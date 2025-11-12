import torch
import torch.nn as nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock, Qwen3MoeConfig
from megablocks import dMoE, Arguments
from megablocks.layers import common
from typing import Tuple
import torch.distributed as dist
import torch.nn.functional as F

# class Qwen3Router(nn.Module):
#     """Custom router that matches Qwen3's exact routing behavior"""
    
#     def __init__(self, args: Arguments, norm_topk_prob: bool):
#         super().__init__()
#         self.args = args
#         self.norm_topk_prob = norm_topk_prob
#         self.layer = nn.Linear(
#             args.hidden_size,
#             args.moe_num_experts,
#             bias=False,
#             dtype=common.dtype(args),
#             device=args.device,
#         )
    
#     def forward(self, x: torch.Tensor):
#         # Compute router logits
#         logits = self.layer(x.view(-1, x.shape[-1]))
        
#         # KEY: Use float32 for softmax like Qwen3 (line 233 of modeling_qwen3_moe.py)
#         # Note: Qwen3 uses dim=1, but since we're viewing as 2D, dim=-1 is equivalent
#         scores = F.softmax(logits, dim=-1, dtype=torch.float32)
        
#         # Get top-k experts (line 234)
#         expert_weights, expert_indices = torch.topk(scores, self.args.moe_top_k, dim=-1)
        
#         # KEY: Qwen3 conditionally normalizes based on norm_topk_prob (lines 235-236)
#         if self.norm_topk_prob:
#             expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
#         # Cast back to input dtype (line 238)
#         expert_weights = expert_weights.to(x.dtype)
        
#         return scores, expert_weights, expert_indices

class Qwen3MegaBlocksAdapter(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, module: Qwen3MoeSparseMoeBlock, ep_group: dist.ProcessGroup):
        super().__init__()

        ep_rank = dist.get_rank(ep_group)
        ep_size = dist.get_world_size(ep_group)

        assert config.num_experts % ep_size == 0, "Number of experts must be divisible by the number of expert parallel groups"
        num_experts_per_rank = config.num_experts // ep_size

        args = Arguments(
            mlp_type="glu",
            mlp_impl="grouped",
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.moe_intermediate_size,
            moe_num_experts=config.num_experts,
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

        # self.moe.router = Qwen3Router(args, norm_topk_prob=config.norm_topk_prob)

        if module is not None:
            with torch.no_grad():
                self.moe.router.layer.weight.copy_(module.gate.weight.clone().detach())
                if module.gate.bias is not None:
                    self.moe.router.layer.bias.copy_(module.gate.bias.clone().detach())

                w1, w2, v1 = [], [], []
                for i in range(num_experts_per_rank):
                    w1.append(module.experts[ep_rank * num_experts_per_rank + i].gate_proj.weight.clone().detach())
                    w2.append(module.experts[ep_rank * num_experts_per_rank + i].down_proj.weight.t().clone().detach())
                    v1.append(module.experts[ep_rank * num_experts_per_rank + i].up_proj.weight.clone().detach())

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
