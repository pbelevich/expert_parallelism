import torch
import torch.distributed as dist
from my_ddp import DataParallelNaive
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

def copy_weights_from(ep_model, model):
    with torch.no_grad():
        for (name1, param1), (name2, param2) in zip(ep_model.named_parameters(), model.named_parameters()):
            assert name1 == f"module.{name2}", f"Parameter names don't match: {name1} vs {name2}"
            param2.copy_(param1)
    return ep_model

def create_batch(config, batch_size=32, seq_len=32):
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device='cuda')
    return input_ids

def assert_all_different_across_ranks(input_ids, world_size, rank):
    all_input_ids = [torch.zeros_like(input_ids) for _ in range(world_size)]
    dist.all_gather(all_input_ids, input_ids)
    for i in range(world_size):
        if i != rank:
            assert not torch.allclose(input_ids, all_input_ids[i])
    print(f"Rank {rank:02d}: All input ids are all different")

def sync_batch(input_ids):
    dist.broadcast(input_ids, src=0)
    return input_ids

def assert_the_same_across_ranks(outputs, world_size, rank, msg=None):
    all_outputs = [torch.zeros_like(outputs) for _ in range(world_size)]
    dist.all_gather(all_outputs, outputs)
    for i in range(world_size):
        torch.testing.assert_close(all_outputs[i], all_outputs[0], msg=msg)
    print(f"Rank {rank:02d}: All outputs are the same")

def assert_outputs_are_all_the_same(outputs, ep_outputs, rank, atol=1e-9, rtol=1e-9):
    assert outputs.logits is not None
    assert ep_outputs.logits is not None
    torch.testing.assert_close(outputs.logits, ep_outputs.logits, atol=atol, rtol=rtol)
    assert outputs.loss is not None
    assert ep_outputs.loss is not None
    torch.testing.assert_close(outputs.loss, ep_outputs.loss, atol=atol, rtol=rtol)
    assert outputs.hidden_states is not None
    assert ep_outputs.hidden_states is not None
    torch.testing.assert_close(outputs.hidden_states, ep_outputs.hidden_states, atol=atol, rtol=rtol)
    
    print(f"Rank {rank:02d}: EP model outputs are the same as the original model outputs")

def assert_gradients_are_the_same_across_ranks(model, world_size, rank, except_patterns=None):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if except_patterns is not None and any(except_name in name for except_name in except_patterns):
                continue
            gradients = param.grad
            assert gradients is not None, "Try to increase number of tokens(batch_size * seq_len)"
            assert_the_same_across_ranks(gradients, world_size, rank, msg=f"Tensor-likes are not close! Rank {rank:02d}: {name=}")
    print(f"Rank {rank:02d}: All gradients are the same")

def apply_expert_parallelism(model, ep_group, impl="mega_blocks"):

    def replace_module(module, old_class, new_class_fn, config, device, dtype, ep_group):
        for name, child in list(module.named_children()):
            replace_module(child, old_class, new_class_fn, config, device, dtype, ep_group)
            if isinstance(child, old_class):
                adapter = new_class_fn(config, ep_group).to(dtype)
                adapter.copy_weights_from(child)
                setattr(module, name, adapter.to(device))
        return module

    if isinstance(model, DataParallelNaive):
        _, grad_atol, grad_rtol, except_patterns, assert_gradients_are_all_the_same = apply_expert_parallelism(model.module, ep_group, impl)
        return model, grad_atol, grad_rtol, except_patterns, assert_gradients_are_all_the_same
    if isinstance(model, MixtralForCausalLM):
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
        if impl == "mega_blocks":
            from mixtral_megablocks import MixtralMegaBlocksAdapter, assert_gradients_are_all_the_same
            model = replace_module(model, MixtralSparseMoeBlock, MixtralMegaBlocksAdapter, model.config, model.device, model.dtype, ep_group)
            grad_atol = 1e-6
            grad_rtol = 1e-6
            except_patterns = ["moe.experts"]
        elif impl == "my":
            from mixtral_my import MixtralSparseMoeBlockEP, assert_gradients_are_all_the_same
            model = replace_module(model, MixtralSparseMoeBlock, MixtralSparseMoeBlockEP, model.config, model.device, model.dtype, ep_group)
            grad_atol = 1e-6
            grad_rtol = 1e-6
            except_patterns = [".experts"]
            assert_gradients_are_all_the_same
        return model, grad_atol, grad_rtol, except_patterns, assert_gradients_are_all_the_same
    elif isinstance(model, Qwen3MoeForCausalLM):
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
        if impl == "mega_blocks":
            from qwen3 import Qwen3MegaBlocksAdapter
            model = replace_module(model, Qwen3MoeSparseMoeBlock, Qwen3MegaBlocksAdapter, model.config, model.device, model.dtype, ep_group)
            grad_atol = 0.2
            grad_rtol = 1e-2
            except_patterns = ["moe.experts"]
        elif impl == "my":
            from qwen3 import Qwen3MoeSparseMoeBlockEP
            model = replace_module(model, Qwen3MoeSparseMoeBlock, Qwen3MoeSparseMoeBlockEP, model.config, model.device, model.dtype, ep_group)
            grad_atol = 1e-6
            grad_rtol = 1e-6
            except_patterns = [".experts"]
        return model, grad_atol, grad_rtol, except_patterns
    elif isinstance(model, Llama4ForCausalLM):
        from transformers.models.llama4.modeling_llama4 import Llama4TextMoe
        from llama4 import Llama4MegaBlocksAdapter
        model = replace_module(model, Llama4TextMoe, Llama4MegaBlocksAdapter, model.config, model.device, model.dtype, ep_group)
        grad_atol = 0.2
        grad_rtol = 1e-3
        except_patterns = ["moe.experts"]
        return model, grad_atol, grad_rtol, except_patterns
    elif isinstance(model, DeepseekV3ForCausalLM):
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
        from deepseek import DeepSeekMegaBlocksAdapter
        model = replace_module(model, DeepseekV3MoE, DeepSeekMegaBlocksAdapter, model.config, model.device, model.dtype, ep_group)
        grad_atol = 0.05
        grad_rtol = 0.1
        except_patterns = ["moe.experts"]
        return model, grad_atol, grad_rtol, except_patterns
    else:
        raise ValueError(f"Unsupported model: {type(model)}")
