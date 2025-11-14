import torch
import torch.nn as nn
from transformers.models.llama4.modeling_llama4 import Llama4TextMoe, Llama4TextConfig, Llama4Router, Llama4TextMLP
from transformers.models.llama4.modeling_llama4 import ACT2FN
from megablocks import dMoE, Arguments
from megablocks.layers import common
from typing import Tuple
import torch.distributed as dist
import torch.nn.functional as F
from ep import dispatch, combine

class Llama4MegaBlocksRouter(nn.Module):
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
    def __init__(self, config: Llama4TextConfig, ep_group: dist.ProcessGroup):
        super().__init__()

        self.config = config

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
            moe_normalize_expert_weights=None,
            activation_fn=F.silu,
            bias=False,
            shared_expert=True,
            fc_kwargs={"dtype": torch.bfloat16},
            shared_expert_hidden_size=config.intermediate_size,
        )

        self.moe = dMoE(args)

        self.moe.router = Llama4MegaBlocksRouter(args)

    def copy_weights_from(self, module: Llama4TextMoe):
        with torch.no_grad():
            self.moe.router.layer.weight.copy_(module.router.weight.clone().detach())
            if module.router.bias is not None:
                self.moe.router.layer.bias.copy_(module.router.bias.clone().detach())

            w1 = module.experts.gate_up_proj[self.ep_rank * self.num_experts_per_rank:(self.ep_rank + 1) * self.num_experts_per_rank,:,:self.config.intermediate_size].clone().detach()
            w2 = module.experts.gate_up_proj[self.ep_rank * self.num_experts_per_rank:(self.ep_rank + 1) * self.num_experts_per_rank,:,self.config.intermediate_size:].clone().detach()
            v1 = module.experts.down_proj[self.ep_rank * self.num_experts_per_rank:(self.ep_rank + 1) * self.num_experts_per_rank,:].clone().detach()

            w1 = w1.transpose(2, 1).reshape(-1, self.config.hidden_size)
            w2 = w2.transpose(2, 1).reshape(-1, self.config.hidden_size)
            v1 = v1.transpose(2, 1).reshape(-1, self.config.hidden_size)

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


class Llama4TextMoeEP(nn.Module):

    def __init__(self, config: Llama4TextConfig, ep_group: dist.ProcessGroup):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.experts = Llama4TextExpertsEP(config, ep_group)
        self.router = Llama4Router(config)
        self.shared_expert = Llama4TextMLP(config)
        self.ep_group = ep_group
        def all_reduce_grad(grad, group=self.ep_group):
            if grad is not None:
                dist.all_reduce(grad, group=group)
            return grad
        self.router.weight.register_hook(all_reduce_grad)
        self.shared_expert.gate_proj.weight.register_hook(all_reduce_grad)
        self.shared_expert.up_proj.weight.register_hook(all_reduce_grad)
        self.shared_expert.down_proj.weight.register_hook(all_reduce_grad)

    def copy_weights_from(self, module: Llama4TextMoe):
        with torch.no_grad():
            # Copy router weights
            self.router.weight.copy_(module.router.weight.clone().detach())
            if hasattr(module.router, 'bias') and module.router.bias is not None:
                self.router.bias.copy_(module.router.bias.clone().detach())
            
            # Copy expert weights for this EP rank
            ep_rank = self.experts.ep_rank
            num_experts_per_rank = self.experts.num_experts_per_rank
            
            # Slice the experts for this rank
            start_idx = ep_rank * num_experts_per_rank
            end_idx = (ep_rank + 1) * num_experts_per_rank
            
            self.experts.gate_up_proj.copy_(
                module.experts.gate_up_proj[start_idx:end_idx].clone().detach()
            )
            self.experts.down_proj.copy_(
                module.experts.down_proj[start_idx:end_idx].clone().detach()
            )
            
            # Copy shared expert weights
            self.shared_expert.gate_proj.weight.copy_(
                module.shared_expert.gate_proj.weight.clone().detach()
            )
            self.shared_expert.up_proj.weight.copy_(
                module.shared_expert.up_proj.weight.clone().detach()
            )
            self.shared_expert.down_proj.weight.copy_(
                module.shared_expert.down_proj.weight.clone().detach()
            )

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_scores, router_logits = self.router(hidden_states)
        
        # Llama4 uses ALL experts with sigmoid scores, not top-k selection
        # Create expert assignments: each token goes to all experts
        num_tokens = hidden_states.shape[0]
        
        # Expand: for each token, create assignments to all experts
        # selected_experts shape: [num_tokens, num_experts] with values 0,1,2...num_experts-1
        selected_experts = torch.arange(self.num_experts, device=hidden_states.device).unsqueeze(0).expand(num_tokens, -1)
        
        # routing_weights is the full router_scores (sigmoid scores for all experts)
        routing_weights = router_scores
        
        # Dispatch tokens to appropriate expert ranks
        dispatched_in, send_split_sizes, received_split_sizes, received_tpe = dispatch(
            hidden_states, selected_experts, self.num_experts, self.experts.ep_group
        )
        
        # Extract the routing weights for each dispatched token
        # For Llama4, each token goes to all experts, so we need to extract the weight for each (token, expert) pair
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        
        dispatched_weights_list = []
        ep_rank = self.experts.ep_rank
        ep_size = self.experts.ep_size
        num_experts_per_rank = self.num_experts // ep_size
        
        for r in range(ep_size):
            for local_expert_idx in range(num_experts_per_rank):
                expert_idx = r * num_experts_per_rank + local_expert_idx
                idx, top_x = torch.where(expert_mask[expert_idx])
                # Extract the routing weights for this expert
                expert_weights = routing_weights[top_x, idx]  # Get weights for these (token, expert) pairs
                dispatched_weights_list.append(expert_weights)
        
        dispatched_weights = torch.cat(dispatched_weights_list)
        
        # Exchange the weights to match the dispatched tokens
        from megablocks.layers.all_to_all import all_to_all
        dispatched_weights, _ = all_to_all(
            dispatched_weights.unsqueeze(-1), 
            received_split_sizes, 
            send_split_sizes,
            self.experts.ep_group
        )
        dispatched_weights = dispatched_weights.squeeze(-1)
        
        # Apply routing weights BEFORE expert computation (like baseline does)
        dispatched_in = dispatched_in * dispatched_weights.unsqueeze(-1)
        
        # Create indices to map the hidden states to the local experts
        local_expert_indices = torch.repeat_interleave(
            torch.arange(received_tpe.shape[1], device=received_tpe.device).repeat(received_tpe.shape[0]), 
            received_tpe.reshape(-1)
        )
        
        routed_out = self.experts(dispatched_in, local_expert_indices)
        
        # Combine results from all expert ranks (weights already applied, so pass None)
        final_routed_out = torch.zeros_like(hidden_states)
        final_routed_out = combine(
            final_routed_out, routed_out, selected_experts, None, 
            send_split_sizes, received_split_sizes, self.num_experts, self.experts.ep_group
        )

        out = self.shared_expert(hidden_states)
        out.add_(final_routed_out)
        return out, router_logits


class Llama4TextExpertsEP(nn.Module):

    def __init__(self, config: Llama4TextConfig, ep_group: dist.ProcessGroup):
        super().__init__()
        self.ep_group = ep_group
        self.ep_rank = dist.get_rank(self.ep_group)
        self.ep_size = dist.get_world_size(self.ep_group)
        assert config.num_local_experts % self.ep_size == 0, "Number of experts must be divisible by the number of expert parallel groups"
        self.num_experts_per_rank = config.num_local_experts // self.ep_size
        
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts_per_rank, self.hidden_size, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts_per_rank, self.expert_dim, self.hidden_size)))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor, local_expert_indices: torch.Tensor) -> torch.Tensor:
        """
        This should really not be run on a single machine, as we are reaching compute bound:
        - the inputs are expected to be "sorted" per expert already.
        - the weights are viewed with another dim, to match num_expert, 1, shape * num_tokens, shape

        Args:
            hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
            local_expert_indices (torch.Tensor): indices mapping tokens to local experts
        Returns:
            torch.Tensor
        """
        # If local_expert_indices provided, process per-expert
        output = torch.empty_like(hidden_states)
        for local_expert_idx in range(self.num_experts_per_rank):
            mask = local_expert_indices == local_expert_idx
            expert_hidden_states = hidden_states[mask]
            if expert_hidden_states.shape[0] > 0:
                gate_up = torch.matmul(expert_hidden_states, self.gate_up_proj[local_expert_idx])
                gate, up = gate_up.chunk(2, dim=-1)
                next_states = torch.matmul((up * self.act_fn(gate)), self.down_proj[local_expert_idx])
                output[mask] = next_states
        return output
