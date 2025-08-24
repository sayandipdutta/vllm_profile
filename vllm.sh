# source .env

# Set variables
GPU_ID=0 # or 0,1 for multi-GPU
PORT=8000
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=256
MODEL_PATH="hfl/Qwen2.5-VL-3B-Instruct-GPTQ-Int4"
GPU_MEMORY_UTILIZATION=0.9 # default 0.9
LOG_FILE="/home/sayan/projects/vllmdemo/vllm_docker.log"

dtype=bfloat16

# Calculate tensor parallel size based on the number of GPUs
# IFS=',' read -r -a GPU_ARRAY <<<"$GPU_ID"
# TENSOR_PARALLEL_SIZE=${#GPU_ARRAY[@]}

# --served-model-name $SERVED_MODEL_NAME \
# --limit-mm-per-prompt '{image:2}' \

# Run the Docker container
docker run --runtime nvidia --gpus "device=0" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /tmp/webzer/assets:/tmp/webzer/assets \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model hfl/Qwen2.5-VL-3B-Instruct-GPTQ-Int4 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --max-model-len 32768 \
    --swap-space 0 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --allowed-local-media-path "/tmp/webzer/assets/IMAGE" \
    2>&1
