# Expert Parallelism

Models:

* Mixtral
    * https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py
        * https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/blob/main/config.json - 8 experts
        * https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1/blob/main/config.json - 8 experts
* Qwen 3
    * https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
        * https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/main/config.json - 128 experts
        * https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507/blob/main/config.json - 128 experts
* Llama4
    * https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py
        * https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/blob/main/config.json - 16 experts
        * https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct/blob/main/config.json - 128 experts
* DeepSeek
    * https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py
        * https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/blob/main/config.json - 256 experts
* gptoss
    * https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py
        * https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json - 32 experts
        * https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json - 128 experts

grouped_gemm:

```bash
pip install git+https://github.com/tgale96/grouped_gemm@main
```

```bash
git clone git@github.com:perplexityai/pplx-kernels.git && cd pplx-kernels && git checkout 12cecfda252e4e646417ac263d96e994d476ee5d
git clone git@github.com:pbelevich/DeepEP.git && cd DeepEP && git checkout 27e8e661857499068275dbaa09e4c15d67d51f81
```

## Docker

```bash
docker build --progress=plain -f ./ep.Dockerfile -t ep .
```

```bash
enroot import -o ./ep.sqsh dockerd://ep:latest
```