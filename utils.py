import torch
import torch.distributed as dist
from transformers import AutoConfig, MixtralForCausalLM, Qwen3MoeForCausalLM, Llama4ForCausalLM, DeepseekV3ForCausalLM
from transformers.models.llama4.configuration_llama4 import Llama4Config
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

def create_config(model_name: str):
    config = AutoConfig.from_pretrained(model_name)
    if isinstance(config, Llama4Config):
        config = config.text_config
    if isinstance(config, DeepseekV3Config):
        config.first_k_dense_replace = 0
        config.moe_intermediate_size = 512  # Set MoE intermediate size explicitly
    config.num_hidden_layers = 1
    config.hidden_size = 128
    config.intermediate_size = 512
    return config

def sync_model_parameters(model):
    print(f"Rank {dist.get_rank()}: Syncing model parameters")
    with torch.no_grad():
        for param in model.parameters():
            dist.broadcast(param, src=0)
    print(f"Rank {dist.get_rank()}: Model parameters synced")
    return model

def create_batch(config, batch_size=32, seq_len=32):
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device='cuda')
    return input_ids

def sync_batch(input_ids):
    dist.broadcast(input_ids, src=0)
    return input_ids

def assert_logits_are_the_same(logits, world_size, rank):
    all_logits = [torch.zeros_like(logits) for _ in range(world_size)]
    dist.all_gather(all_logits, logits)
    for i in range(world_size):
        torch.testing.assert_close(all_logits[i], all_logits[0])
    print(f"Rank {rank:02d}: All logits are the same")

def assert_loss_are_the_same(loss, world_size, rank):
    all_loss = [torch.zeros_like(loss) for _ in range(world_size)]
    dist.all_gather(all_loss, loss)
    for i in range(world_size):
        torch.testing.assert_close(all_loss[i], all_loss[0])
    print(f"Rank {rank:02d}: All loss are the same")

def assert_gradients_are_the_same(model, world_size, rank, except_patterns=[]):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(except_name in name for except_name in except_patterns):
                continue
            gradients = param.grad
            assert gradients is not None, "Try to increase number of tokens(batch_size * seq_len)"
            all_gradients = [torch.zeros_like(gradients) for _ in range(world_size)]
            dist.all_gather(all_gradients, gradients)
            for i in range(world_size):
                torch.testing.assert_close(all_gradients[i], all_gradients[0])
    print(f"Rank {rank:02d}: All gradients are the same")

def apply_expert_parallelism(model, ep_group):

    def replace_module(module, old_class, new_class_fn, config, device, ep_group):
        for name, child in list(module.named_children()):
            replace_module(child, old_class, new_class_fn, config, device, ep_group)
            if isinstance(child, old_class):
                adapter = new_class_fn(config, ep_group)
                adapter.copy_weights_from(child)
                setattr(module, name, adapter.to(device))
        return module

    if isinstance(model, MixtralForCausalLM):
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
        from mixtral import MixtralMegaBlocksAdapter
        model = replace_module(model, MixtralSparseMoeBlock, MixtralMegaBlocksAdapter, model.config, model.device, ep_group)
        model.grad_atol = 1e-2
        model.grad_rtol = 1e-2
        return model
    elif isinstance(model, Qwen3MoeForCausalLM):
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
        from qwen3 import Qwen3MegaBlocksAdapter
        model = replace_module(model, Qwen3MoeSparseMoeBlock, Qwen3MegaBlocksAdapter, model.config, model.device, ep_group)
        model.grad_atol = 0.2
        model.grad_rtol = 1e-2
        return model
    elif isinstance(model, Llama4ForCausalLM):
        from transformers.models.llama4.modeling_llama4 import Llama4TextMoe
        from llama4 import Llama4MegaBlocksAdapter
        model = replace_module(model, Llama4TextMoe, Llama4MegaBlocksAdapter, model.config, model.device, ep_group)
        model.grad_atol = 0.2
        model.grad_rtol = 1e-3
        return model
    elif isinstance(model, DeepseekV3ForCausalLM):
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
        from deepseek import DeepSeekMegaBlocksAdapter
        model = replace_module(model, DeepseekV3MoE, DeepSeekMegaBlocksAdapter, model.config, model.device, ep_group)
        model.grad_atol = 0.05
        model.grad_rtol = 0.1
        return model
    else:
        raise ValueError(f"Unsupported model: {type(model)}")
