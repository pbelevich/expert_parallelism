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


class MixtralSparseMoeBlockEP(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accommodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config: MixtralConfig, ep_group: dist.ProcessGroup):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        self.ep_group = ep_group
        self.ep_rank = dist.get_rank(self.ep_group)
        self.ep_size = dist.get_world_size(self.ep_group)
        assert config.num_local_experts % self.ep_size == 0, "Number of experts must be divisible by the number of expert parallel groups"
        self.num_experts_per_rank = config.num_local_experts // self.ep_size

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        def all_reduce_gate_grad(grad, group=self.ep_group):
            if grad is not None:
                dist.all_reduce(grad, group=group)
            return grad
        self.gate.weight.register_hook(all_reduce_gate_grad)
        
        self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP(config) for _ in range(self.num_experts_per_rank)])

        # Jitter parameters
        self.jitter_noise = config.router_jitter_noise

    def copy_weights_from(self, module: MixtralSparseMoeBlock):
        with torch.no_grad():
            self.gate.weight.copy_(module.gate.weight.clone().detach())
            if module.gate.bias is not None:
                self.gate.bias.copy_(module.gate.bias.clone().detach())

            for i in range(self.num_experts_per_rank):
                self.experts[i].w1.weight.copy_(module.experts[self.ep_rank * self.num_experts_per_rank + i].w1.weight.clone().detach())
                self.experts[i].w2.weight.copy_(module.experts[self.ep_rank * self.num_experts_per_rank + i].w2.weight.clone().detach())
                self.experts[i].w3.weight.copy_(module.experts[self.ep_rank * self.num_experts_per_rank + i].w3.weight.clone().detach())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        #################################################################################

        hidden_states, send_split_sizes, received_split_sizes, received_tpe = dispatch(hidden_states, selected_experts, self.num_experts, self.ep_group)

        # Create indices to map the hidden states to the local experts
        local_expert_indices = torch.repeat_interleave(torch.arange(received_tpe.shape[1], device=received_tpe.device).repeat(received_tpe.shape[0]), received_tpe.reshape(-1))

        # Process the hidden states for each local expert
        for local_expert_idx in range(self.num_experts_per_rank):
            expert_layer = self.experts[local_expert_idx]
            expert_hidden_states = hidden_states[local_expert_indices == local_expert_idx]
            hidden_states[local_expert_indices == local_expert_idx] = expert_layer(expert_hidden_states)

        final_hidden_states = combine(final_hidden_states, hidden_states, selected_experts, routing_weights, send_split_sizes, received_split_sizes, self.num_experts, self.ep_group)

        #################################################################################

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
