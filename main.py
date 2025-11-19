import argparse
import copy
import torch
import torch.distributed as dist
from my_ddp import DataParallelNaive
from transformers import AutoModelForCausalLM
from utils import *

def run(rank, local_rank, world_size, args):
    print(f"Hello from rank {rank:02d} of {world_size:02d} on device {local_rank:02d}")
    config = create_config(args.model_name)
    ep_group = dist.group.WORLD
    
    model = sync_model_parameters(AutoModelForCausalLM.from_config(config).to('cuda'))
    ep_model = copy.deepcopy(model)
    model = DataParallelNaive(model, process_group=dist.group.WORLD)
    ep_model = DataParallelNaive(ep_model, process_group=dist.group.WORLD, except_names=["experts"])
    ep_model, grad_atol, grad_rtol, except_patterns, assert_gradients_are_all_the_same = apply_expert_parallelism(ep_model, dist.group.WORLD, args.impl)

    input_ids = sync_batch(create_batch(config))
    outputs = model(input_ids=input_ids, labels=input_ids, output_hidden_states=True)
    assert_the_same_across_ranks(outputs.logits, world_size, rank)
    assert_the_same_across_ranks(outputs.loss, world_size, rank)
    outputs.loss.backward()
    assert_gradients_are_the_same_across_ranks(model, world_size, rank)

    ep_outputs = ep_model(input_ids=input_ids, labels=input_ids, output_hidden_states=True)
    assert_the_same_across_ranks(ep_outputs.logits, world_size, rank)
    assert_the_same_across_ranks(ep_outputs.loss, world_size, rank)
    ep_outputs.loss.backward()
    assert_gradients_are_the_same_across_ranks(ep_model, world_size, rank, except_patterns=except_patterns)

    assert_outputs_are_all_the_same(outputs, ep_outputs, rank, atol=grad_atol, rtol=grad_rtol)

    # DP -> EP -> DP test
    model = sync_model_parameters(AutoModelForCausalLM.from_config(config).to('cuda'))
    ep_model = copy.deepcopy(model)
    model = DataParallelNaive(model, process_group=dist.group.WORLD)
    ep_model = DataParallelNaive(ep_model, process_group=dist.group.WORLD, except_names=["experts"])
    ep_model, grad_atol, grad_rtol, except_patterns, assert_gradients_are_all_the_same = apply_expert_parallelism(ep_model, dist.group.WORLD, args.impl)

    input_ids = create_batch(config)
    assert_all_different_across_ranks(input_ids, world_size, rank)
    outputs = model(input_ids=input_ids, labels=input_ids, output_hidden_states=True)
    ep_outputs = ep_model(input_ids=input_ids, labels=input_ids, output_hidden_states=True)
    assert_outputs_are_all_the_same(outputs, ep_outputs, rank)
    outputs.loss.backward()
    ep_outputs.loss.backward()
    assert_gradients_are_all_the_same(model, ep_model, ep_group, rank)


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

SUPPORTED_MODELS = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "deepseek-ai/DeepSeek-R1-0528",
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1", choices=SUPPORTED_MODELS)
    parser.add_argument("--impl", type=str, default="mega_blocks", choices=["mega_blocks", "my"])
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
    print("Done")
