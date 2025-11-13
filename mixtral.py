import torch
import torch.nn as nn
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock, MixtralConfig
from megablocks import dMoE, Arguments
from megablocks.layers import common
from typing import Tuple
import torch.distributed as dist
import torch.nn.functional as F

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
