import torch
import torch.nn as nn
from transformers.models.llama4.modeling_llama4 import Llama4TextMoe, Llama4TextConfig
from megablocks import dMoE, Arguments
from megablocks.layers import common
from typing import Tuple
import torch.distributed as dist
import torch.nn.functional as F

class Llama4Router(nn.Module):
    """Custom router that matches Llama4's unique sigmoid-based routing"""
    
    def __init__(self, args: Arguments):
        super().__init__()
        self.args = args
        self.num_experts = args.moe_num_experts
        self.top_k = args.moe_top_k
        self.layer = nn.Linear(
            args.hidden_size,
            args.moe_num_experts,
            bias=False,
            dtype=common.dtype(args),
            device=args.device,
        )
    
    def forward(self, x: torch.Tensor):
        logits = self.layer(x.view(-1, x.shape[-1]))
        router_top_value, router_indices = torch.topk(logits, self.top_k, dim=1)
        router_scores = torch.full_like(logits, float("-inf")).scatter_(1, router_indices, router_top_value)
        router_scores = F.sigmoid(router_scores.float()).to(logits.dtype)
        expert_weights = router_scores.gather(1, router_indices)  
        return router_scores, expert_weights, router_indices


class Llama4MegaBlocksAdapter(nn.Module):
    def __init__(self, config: Llama4TextConfig, module: Llama4TextMoe, ep_group: dist.ProcessGroup):
        super().__init__()

        ep_rank = dist.get_rank(ep_group)
        ep_size = dist.get_world_size(ep_group)

        assert config.num_local_experts % ep_size == 0, "Number of experts must be divisible by the number of expert parallel groups"
        num_experts_per_rank = config.num_local_experts // ep_size

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
            moe_normalize_expert_weights=None,
            activation_fn=F.silu,
            bias=False,
            shared_expert=True,
            fc_kwargs={"dtype": torch.bfloat16},
            shared_expert_hidden_size=config.intermediate_size,
        )

        self.moe = dMoE(args)

        self.moe.router = Llama4Router(args)

        if module is not None:
            with torch.no_grad():
                self.moe.router.layer.weight.copy_(module.router.weight.clone().detach())
                if module.router.bias is not None:
                    self.moe.router.layer.bias.copy_(module.router.bias.clone().detach())

                w1 = module.experts.gate_up_proj[ep_rank * num_experts_per_rank:(ep_rank + 1) * num_experts_per_rank,:,:config.intermediate_size].clone().detach()
                w2 = module.experts.gate_up_proj[ep_rank * num_experts_per_rank:(ep_rank + 1) * num_experts_per_rank,:,config.intermediate_size:].clone().detach()
                v1 = module.experts.down_proj[ep_rank * num_experts_per_rank:(ep_rank + 1) * num_experts_per_rank,:].clone().detach()

                w1 = w1.transpose(2, 1).reshape(-1, config.hidden_size)
                w2 = w2.transpose(2, 1).reshape(-1, config.hidden_size)
                v1 = v1.transpose(2, 1).reshape(-1, config.hidden_size)

                self.moe.experts.mlp.w1.copy_(w1)
                self.moe.experts.mlp.w2.copy_(w2)
                self.moe.experts.mlp.v1.copy_(v1)

                self.moe.shared_expert.gate_proj.weight.copy_(
                    module.shared_expert.gate_proj.weight.clone().detach()
                )
                self.moe.shared_expert.up_proj.weight.copy_(
                    module.shared_expert.up_proj.weight.clone().detach()
                )
                self.moe.shared_expert.down_proj.weight.copy_(
                    module.shared_expert.down_proj.weight.clone().detach()
                )


    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # HF: (B, S, H); MegaBlocks: (S, B, H)
        x = hidden_states.transpose(0, 1).contiguous()  # (S, B, H)
        out = self.moe(x)  # out: (S, B, H)
        out = out.transpose(0, 1).contiguous()  # back to (B, S, H)

        # MegaBlocks keeps router / load-balancing losses in global buffers, so we can
        # just return None here – training code will read them separately.
        router_logits = None
        return out, router_logits
