import argparse
import torch
import torch.distributed as dist
from megablocks.layers.all_to_all import all_to_all

def dispatch(hidden_states, selected_experts, num_experts, ep_group, print_debug=False):
    ep_rank = dist.get_rank(ep_group)
    ep_size = dist.get_world_size(ep_group)
    num_experts_per_rank = num_experts // ep_size

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)

    # print(f"Rank {rank:2d}\n{expert_mask=}\n")

    send_tpe = torch.empty(ep_size, num_experts_per_rank, dtype=torch.int32, device="cuda")
        
    # List of all hidden states to send
    hidden_states_to_send = []

    for r in range(ep_size):
        for local_expert_idx in range(num_experts_per_rank):
            expert_idx = r * num_experts_per_rank + local_expert_idx
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[None, top_x].reshape(-1, hidden_states.shape[-1])
            hidden_states_to_send.append(current_state)
            send_tpe[r, local_expert_idx] = current_state.shape[0]

    # Concatenate all hidden states to send
    hidden_states_to_send = torch.cat(hidden_states_to_send, dim=0)

    if print_debug:
        print(f"Rank {ep_rank:2d}\n{hidden_states_to_send=}\n")
        print(f"Rank {ep_rank:2d}\n{send_tpe=}\n")

    # Holds the number of tokens to receive from each ep_rank
    received_tpe = torch.empty_like(send_tpe)

    # Exchange the number of tokens to send to each ep_rank to get the number of tokens to receive for this rank
    dist.all_to_all_single(received_tpe, send_tpe, group=ep_group)

    if print_debug:
        print(f"Rank {ep_rank:2d}\n{received_tpe=}\n")

    # Holds the number of tokens to send to each ep_rank
    send_split_sizes = send_tpe.sum(dim=1).tolist()

    # Holds the number of tokens to receive from each ep_rank
    received_split_sizes = received_tpe.sum(dim=1).tolist()

    # Exchange the hidden states to send to each ep_rank to get the hidden states for this rank
    hidden_states_received, _ = all_to_all(hidden_states_to_send, received_split_sizes, send_split_sizes, ep_group)

    if print_debug:
        print(f"Rank {ep_rank:2d}\n{hidden_states_received=}\n")

    return hidden_states_received, send_split_sizes, received_split_sizes, received_tpe

def combine(final_hidden_states, hidden_states, selected_experts, routing_weights, send_split_sizes, received_split_sizes, num_experts, ep_group, print_debug=False):
    ep_size = dist.get_world_size(ep_group)
    num_experts_per_rank = num_experts // ep_size

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)

    # Exchange the hidden states back to the original rank with gradients
    new_hidden_states_to_send, _ = all_to_all(hidden_states, send_split_sizes, received_split_sizes, ep_group)

    # Reconstruct using the original expert mask and routing weights
    offset = 0
    for rank in range(ep_size):
        for local_expert_idx in range(num_experts_per_rank):
            expert_idx = rank * num_experts_per_rank + local_expert_idx
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_hidden_states = new_hidden_states_to_send[offset:offset + len(top_x)]

            # Apply routing weights if provided (None means weights were already applied)
            if routing_weights is not None:
                current_hidden_states = current_hidden_states * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

            offset += len(top_x)

    return final_hidden_states

def run(rank, local_rank, world_size, args):
    print(f"Hello from rank {rank:02d} of {world_size:02d} on device {local_rank:02d}")

    ep_group = dist.group.WORLD
    ep_rank = dist.get_rank(ep_group)
    ep_size = dist.get_world_size(ep_group)
    num_experts_per_rank = 2
    num_experts = num_experts_per_rank * ep_size

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

    hidden_states, send_split_sizes, received_split_sizes, received_tpe = dispatch(hidden_states, selected_experts, num_experts, ep_group, print_debug=True)

    # Create indices to map the hidden states to the local experts
    local_expert_indices = torch.repeat_interleave(torch.arange(received_tpe.shape[1], device=received_tpe.device).repeat(received_tpe.shape[0]), received_tpe.reshape(-1))


    print(f"Rank {ep_rank:2d}\n{local_expert_indices=}\n")

    # Process the hidden states for each local expert
    for local_expert_idx in range(num_experts_per_rank):
        expert_hidden_states = hidden_states[local_expert_indices == local_expert_idx]
        hidden_states[local_expert_indices == local_expert_idx] = expert_hidden_states + 100 * (local_expert_idx + rank * num_experts_per_rank)

    final_hidden_states = torch.zeros(b * s, h, device=hidden_states.device)
    final_hidden_states = combine(final_hidden_states, hidden_states, selected_experts, routing_weights, send_split_sizes, received_split_sizes, num_experts, ep_group, print_debug=True)
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
