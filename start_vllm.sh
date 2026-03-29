#!/bin/bash
# Generic script to start a vLLM server for model inference

model=$1
size=${2:-4}
port=${3:-8000}

if [ -z "$model" ]; then
    echo "Usage: ./start_vllm.sh <model_name_or_path> [tensor_parallel_size] [port]"
    echo "Example: ./start_vllm.sh meta-llama/Llama-3.1-8B-Instruct 4 8000"
    exit 1
fi

# convenient short names
if [ "$model" == "llama3.1-8b" ]; then
    model="meta-llama/Llama-3.1-8B-Instruct" 
elif [ "$model" == "llama3.3-70b" ]; then
    model="meta-llama/Llama-3.3-70B-Instruct"
elif [ "$model" == "qwen3-80b" ]; then
    model="Qwen/Qwen3-Next-80B-A3B-Instruct"
fi

echo "Starting vLLM API server with model $model on port $port with tensor parallel size $size"

# Start the vllm server. Make sure you have installed vllm (`pip install vllm`)
vllm serve $model \
    --tensor-parallel-size $size \
    --port $port \
    --dtype auto \
    --trust-remote-code \
    --api-key "token-abc123" \
    --gpu-memory-utilization 0.9
