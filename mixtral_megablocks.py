import torch
import torch.nn as nn
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock, MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralBlockSparseTop2MLP
from megablocks import dMoE, Arguments
from megablocks.layers import common
from typing import Tuple
import torch.distributed as dist
import torch.nn.functional as F
from ep import dispatch, combine

class MixtralRouter(nn.Module):
    """Custom router that matches Mixtral's float32 softmax behavior"""
    
    def __init__(self, args: Arguments):
        super().__init__()
        self.args = args
        self.layer = nn.Linear(
            args.hidden_size,
            args.moe_num_experts,
            bias=False,
            dtype=common.dtype(args),
            device=args.device,
        )
    
    def forward(self, x: torch.Tensor):
        logits = self.layer(x.view(-1, x.shape[-1]))
        scores = F.softmax(logits, dim=-1, dtype=torch.float32)
        expert_weights, expert_indices = torch.topk(scores, self.args.moe_top_k, dim=-1)
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        expert_weights = expert_weights.to(x.dtype)
        return scores, expert_weights, expert_indices

class MixtralMegaBlocksAdapter(nn.Module):
    def __init__(self, config: MixtralConfig, ep_group: dist.ProcessGroup):
        super().__init__()

        self.ep_rank = dist.get_rank(ep_group)
        self.ep_size = dist.get_world_size(ep_group)

        assert config.num_local_experts % self.ep_size == 0, "Number of experts must be divisible by the number of expert parallel groups"
        self.num_experts_per_rank = config.num_local_experts // self.ep_size

        args = Arguments(
            mlp_type="glu",
            mlp_impl="grouped",
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            moe_num_experts=config.num_local_experts,
            moe_top_k=config.num_experts_per_tok,
            moe_capacity_factor=1,
            moe_expert_model_parallelism=True,
            expert_parallel_group=ep_group,
            bf16=True,
            fp16=False,
            moe_normalize_expert_weights=1.0,
            activation_fn=F.silu,
            bias=False,
        )

        self.moe = dMoE(args)
        self.moe.router = MixtralRouter(args)


    def copy_weights_from(self, module: MixtralSparseMoeBlock):
        with torch.no_grad():
            self.moe.router.layer.weight.copy_(module.gate.weight.clone().detach())
            if module.gate.bias is not None:
                self.moe.router.layer.bias.copy_(module.gate.bias.clone().detach())

            w1, w2, v1 = [], [], []
            for i in range(self.num_experts_per_rank):
                w1.append(module.experts[self.ep_rank * self.num_experts_per_rank + i].w1.weight.clone().detach())
                w2.append(module.experts[self.ep_rank * self.num_experts_per_rank + i].w2.weight.t().clone().detach())
                v1.append(module.experts[self.ep_rank * self.num_experts_per_rank + i].w3.weight.clone().detach())

            self.moe.experts.mlp.w1.copy_(torch.cat(w1, dim=0))
            self.moe.experts.mlp.w2.copy_(torch.cat(w2, dim=0))
            self.moe.experts.mlp.v1.copy_(torch.cat(v1, dim=0))


    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # HF: (B, S, H); MegaBlocks: (S, B, H)
        x = hidden_states.transpose(0, 1).contiguous()  # (S, B, H)
        out = self.moe(x)  # out: (S, B, H)
        out = out.transpose(0, 1).contiguous()  # back to (B, S, H)

        # MegaBlocks keeps router / load-balancing losses in global buffers, so we can
        # just return None here – training code will read them separately.
        router_logits = None
        return out, router_logits

def assert_gradients_are_all_the_same(model, ep_model, ep_group, rank):
    param = model.module.lm_head.weight
    ep_param = ep_model.module.lm_head.weight
    torch.testing.assert_close(param.grad, ep_param.grad, atol=1e-12, rtol=1e-12)
    
    for name, param in model.named_parameters():
        if 'experts' in name or 'gate' in name:
            continue
        ep_param = ep_model.get_parameter(name)
        torch.testing.assert_close(param.grad, ep_param.grad, atol=1e-4, rtol=1e-4)

    num_local_experts = model.module.config.num_local_experts
    ep_size = dist.get_world_size(ep_group)
    
    atol, rtol = 1e-3, 1e-3

    def assert_gradients_are_the_same(name, ep_param_grad):
        all_experts_grads = [getattr(layer.block_sparse_moe.experts[i], name).weight.grad for i in range(num_local_experts)]
        all_experts_grads = torch.cat(all_experts_grads, dim=0)
        all_ep_experts_grads = [torch.zeros_like(ep_param_grad) for _ in range(ep_size)]
        dist.all_gather(all_ep_experts_grads, ep_param_grad)
        all_ep_experts_grads = torch.cat(all_ep_experts_grads, dim=0)
        torch.testing.assert_close(all_experts_grads, all_ep_experts_grads, atol=atol, rtol=rtol)

    
    for layer, ep_layer in zip(model.module.model.layers, ep_model.module.model.layers):
        mlp = ep_layer.block_sparse_moe.moe.experts.mlp
        assert_gradients_are_the_same("w1", mlp.w1.grad)
        assert_gradients_are_the_same("w2", mlp.w2.grad.t().contiguous())
        assert_gradients_are_the_same("w3", mlp.v1.grad)

    print(f"Rank {rank:02d}: All gradients are the same")
