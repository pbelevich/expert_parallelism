import torch
import torch.nn as nn
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from megablocks import dMoE, Arguments
from megablocks.layers import common
import torch.distributed as dist
import torch.nn.functional as F

class DeepSeekMegaBlocksAdapter(nn.Module):
    def __init__(self, config: DeepseekV3Config, ep_group: dist.ProcessGroup):
        super().__init__()

        self.ep_rank = dist.get_rank(ep_group)
        self.ep_size = dist.get_world_size(ep_group)

        assert config.n_routed_experts % self.ep_size == 0, "Number of experts must be divisible by the number of expert parallel groups"
        self.num_experts_per_rank = config.n_routed_experts // self.ep_size

        args = Arguments(
            mlp_type="glu",
            mlp_impl="grouped",
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.moe_intermediate_size,  # Use MoE-specific intermediate size
            moe_num_experts=config.n_routed_experts,
            moe_top_k=config.num_experts_per_tok,
            moe_capacity_factor=1,
            moe_expert_model_parallelism=True,
            expert_parallel_group=ep_group,
            bf16=True,
            fp16=False,
            moe_normalize_expert_weights=1.0,
            activation_fn=F.silu,
            bias=False,
            shared_expert=True,
            fc_kwargs={"dtype": torch.bfloat16},
            shared_expert_hidden_size=config.moe_intermediate_size * config.n_shared_experts,  # Shared expert size
        )

        self.moe = dMoE(args)

    def copy_weights_from(self, module: DeepseekV3MoE):
        with torch.no_grad():
            self.moe.router.layer.weight.copy_(module.gate.weight.clone().detach())

            w1, w2, v1 = [], [], []
            for i in range(self.num_experts_per_rank):
                w1.append(module.experts[self.ep_rank * self.num_experts_per_rank + i].gate_proj.weight.clone().detach())
                w2.append(module.experts[self.ep_rank * self.num_experts_per_rank + i].down_proj.weight.t().clone().detach())
                v1.append(module.experts[self.ep_rank * self.num_experts_per_rank + i].up_proj.weight.clone().detach())

            w1 = torch.cat(w1, dim=0)
            w2 = torch.cat(w2, dim=0)
            v1 = torch.cat(v1, dim=0)

            self.moe.experts.mlp.w1.copy_(w1)
            self.moe.experts.mlp.w2.copy_(w2)
            self.moe.experts.mlp.v1.copy_(v1)

            self.moe.shared_expert.gate_proj.weight.copy_(
                module.shared_experts.gate_proj.weight.clone().detach()
            )
            self.moe.shared_expert.up_proj.weight.copy_(
                module.shared_experts.up_proj.weight.clone().detach()
            )
            self.moe.shared_expert.down_proj.weight.copy_(
                module.shared_experts.down_proj.weight.clone().detach()
            )


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # HF: (B, S, H); MegaBlocks: (S, B, H)
        x = hidden_states.transpose(0, 1).contiguous()  # (S, B, H)
        out = self.moe(x)  # out: (S, B, H)
        out = out.transpose(0, 1).contiguous()  # back to (B, S, H)
        
        # DeepseekV3MoE.forward() returns just a tensor, not a tuple
        return out
