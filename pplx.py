import argparse
import torch
import torch.distributed as dist
from pplx_kernels import AllToAll

class PPLXAllToAll:

    def __init__(self, ep_group: dist.ProcessGroup, num_experts: int, top_k: int, hidden_dim: int, hidden_dim_bytes:int, max_num_tokens:int, use_internode: bool = False):
        self.ep_group = ep_group
        self.ep_rank = dist.get_rank(self.ep_group)
        self.ep_size = dist.get_world_size(self.ep_group)
        self.num_experts = num_experts
        assert num_experts % self.ep_size == 0, "Number of experts must be divisible by the number of expert parallel groups"
        self.num_experts_per_rank = num_experts // self.ep_size
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.max_num_tokens = max_num_tokens

        if use_internode:
            self.all_to_all = AllToAll.internode(
                max_num_tokens=max_num_tokens,
                num_experts=self.num_experts,
                experts_per_token=self.top_k,
                rank=self.ep_rank,
                world_size=self.ep_size,
                dp_size=1,
                hidden_dim=self.hidden_dim,
                hidden_dim_bytes=hidden_dim_bytes,
                hidden_dim_scale_bytes=0,
                group_name=self.ep_group.group_name,
            )
        else:
            self.all_to_all = AllToAll.intranode(
                max_num_tokens=max_num_tokens,
                num_experts=self.num_experts,
                experts_per_token=self.top_k,
                rank=self.ep_rank,
                world_size=self.ep_size,
                dp_size=1,
                hidden_dim=self.hidden_dim,
                hidden_dim_bytes=hidden_dim_bytes,
                hidden_dim_scale_bytes=0,
                group_name=self.ep_group.group_name,
            )

    def dispatch(self, hidden_states, selected_experts, num_tokens_tensor, print_debug=False):
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        received_tpe = torch.empty(self.num_experts_per_rank, dtype=torch.int32, device=hidden_states.device)
        
        hidden_states_received = torch.zeros(
            (self.num_experts_per_rank, self.max_num_tokens, self.hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
                
        self.all_to_all.dispatch(
            out_expert_num_tokens=received_tpe,             # [num_experts_per_rank]
            out_expert_x=hidden_states_received,            # [num_experts_per_rank, max_num_tokens, hidden_dim]
            out_expert_x_scale=None,
            dp_x=hidden_states,                             # [num_tokens, hidden_dim]
            dp_x_scale=None,
            indices=selected_experts.to(torch.uint32),      # [num_tokens, top_k]
            bound_m=num_tokens_tensor,
        )

        return hidden_states_received, received_tpe

    def combine(self, final_hidden_states, hidden_states, selected_experts, routing_weights, bound_m, print_debug=False):
        self.all_to_all.combine(
            out_tokens=final_hidden_states,
            indices=selected_experts,
            weights=routing_weights,
            expert_y=hidden_states,
            bound_m=bound_m,
        )
        return final_hidden_states

def run(rank, local_rank, world_size, args):
    print(f"Hello from rank {rank:02d} of {world_size:02d} on device {local_rank:02d}")

    ep_group = dist.group.WORLD
    ep_rank = dist.get_rank(ep_group)
    ep_size = dist.get_world_size(ep_group)
    num_experts_per_rank = 2
    num_experts = num_experts_per_rank * ep_size

    pplx = PPLXAllToAll()

    hidden_states = torch.tensor([[
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0, 3.0],
    ],[
        [4.0, 4.0, 4.0, 4.0],
        [5.0, 5.0, 5.0, 5.0],
        [6.0, 6.0, 6.0, 6.0],
    ]], device=f"cuda:{local_rank}") + rank * 6

    b, s, h = hidden_states.shape

    hidden_states = hidden_states.view(-1, h)

    if rank == 0:
        selected_experts = torch.tensor([
            [0, 2], 
            [2, 0], 
            [0, 2],
            [3, 2], 
            [1, 0], 
            [0, 1]
        ], device=f"cuda:{local_rank}")
    else:
        selected_experts = torch.tensor([
            [1, 3], 
            [3, 1], 
            [1, 3],
            [2, 3], 
            [0, 1], 
            [1, 0]
        ], device=f"cuda:{local_rank}")

    routing_weights = torch.tensor([
        [0.8, 0.2],
        [0.6, 0.4],
        [0.9, 0.1],
        [0.7, 0.3], 
        [0.6, 0.4],
        [0.8, 0.2],
    ], device=f"cuda:{local_rank}")

    print(f"Rank {ep_rank:2d}\n{hidden_states=}\n")
    print(f"Rank {ep_rank:2d}\n{selected_experts=}\n")

    hidden_states, send_split_sizes, received_split_sizes, received_tpe = pplx.dispatch(hidden_states, selected_experts, num_experts, ep_group, print_debug=True)

    # Create indices to map the hidden states to the local experts
    local_expert_indices = torch.repeat_interleave(torch.arange(received_tpe.shape[1], device=received_tpe.device).repeat(received_tpe.shape[0]), received_tpe.reshape(-1))

    print(f"Rank {ep_rank:2d}\n{local_expert_indices=}\n")

    # Process the hidden states for each local expert
    for local_expert_idx in range(num_experts_per_rank):
        expert_hidden_states = hidden_states[local_expert_indices == local_expert_idx]
        hidden_states[local_expert_indices == local_expert_idx] = expert_hidden_states + 100 * (local_expert_idx + rank * num_experts_per_rank)

    final_hidden_states = torch.zeros(b * s, h, device=hidden_states.device)
    final_hidden_states = pplx.combine(final_hidden_states, hidden_states, selected_experts, routing_weights, send_split_sizes, received_split_sizes, num_experts, ep_group, print_debug=True)
    final_hidden_states = final_hidden_states.reshape(b, s, h)

    print(f"Rank {ep_rank:2d}\n{final_hidden_states=}\n")


def main(args):
    dist.init_process_group(backend="nccl")
    try:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        torch.manual_seed(rank)
        torch.cuda.manual_seed(rank)
        run(rank, local_rank, world_size, args)
    finally:
        dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
    print("Done")
