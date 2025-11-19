import torch
import torch.nn as nn
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock, MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralBlockSparseTop2MLP
import torch.distributed as dist
import torch.nn.functional as F
from ep import dispatch, combine

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

def assert_gradients_are_all_the_same(model, ep_model, ep_group, rank):
    for name, param in model.named_parameters():
        if 'experts' in name:
            continue
        elif 'gate' in name:
            ep_param = ep_model.get_parameter(name)
            torch.testing.assert_close(param.grad, ep_param.grad, atol=1e-2, rtol=1e-2)
        else:
            ep_param = ep_model.get_parameter(name)
            torch.testing.assert_close(param.grad, ep_param.grad, atol=1e-12, rtol=1e-12)

    ep_size = dist.get_world_size(ep_group)

    def assert_gradients_are_the_same(experts, ep_experts, name):
        all_grads = torch.cat([getattr(expert, name).weight.grad for expert in experts], dim=0)
        ep_grads = torch.cat([getattr(expert, name).weight.grad for expert in ep_experts], dim=0)
        all_ep_grads = [torch.zeros_like(ep_grads) for _ in range(ep_size)]
        dist.all_gather(all_ep_grads, ep_grads)
        all_ep_grads = torch.cat(all_ep_grads, dim=0)
        torch.testing.assert_close(all_grads, all_ep_grads, atol=1e-2, rtol=1e-2)
    
    for layer, ep_layer in zip(model.module.model.layers, ep_model.module.model.layers):
        assert_gradients_are_the_same(layer.block_sparse_moe.experts, ep_layer.block_sparse_moe.experts, "w1")
        assert_gradients_are_the_same(layer.block_sparse_moe.experts, ep_layer.block_sparse_moe.experts, "w2")
        assert_gradients_are_the_same(layer.block_sparse_moe.experts, ep_layer.block_sparse_moe.experts, "w3")

    print(f"Rank {rank:02d}: All gradients are the same")
