import argparse
import copy
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM
from utils import *

def run(rank, local_rank, world_size, args):
    print(f"Hello from rank {rank:02d} of {world_size:02d} on device {local_rank:02d}")
    config = create_config(args.model_name)
    model = sync_model_parameters(AutoModelForCausalLM.from_config(config).to('cuda'))
    ep_model = apply_expert_parallelism(copy.deepcopy(model), dist.group.WORLD, args.impl)

    input_ids = sync_batch(create_batch(config))
    outputs = model(input_ids=input_ids, labels=input_ids)
    assert_logits_are_the_same(outputs.logits, world_size, rank)
    assert_loss_are_the_same(outputs.loss, world_size, rank)
    if args.impl != "pplx":
        outputs.loss.backward()
        assert_gradients_are_the_same(model, world_size, rank)

    ep_outputs = ep_model(input_ids=input_ids, labels=input_ids)
    assert_logits_are_the_same(ep_outputs.logits, world_size, rank)
    assert_loss_are_the_same(ep_outputs.loss, world_size, rank)
    if args.impl != "pplx":
        ep_outputs.loss.backward()
        assert_gradients_are_the_same(ep_model, world_size, rank)

    torch.testing.assert_close(outputs.logits, ep_outputs.logits, atol=ep_model.grad_atol, rtol=ep_model.grad_rtol)
    torch.testing.assert_close(outputs.loss, ep_outputs.loss, atol=ep_model.grad_atol, rtol=ep_model.grad_rtol)


def main(args):
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
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
    parser.add_argument("--impl", type=str, default="mega_blocks", choices=["mega_blocks", "my", "pplx"])
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
    print("Done")
