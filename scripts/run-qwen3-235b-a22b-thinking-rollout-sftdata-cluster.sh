#!/bin/bash
set -x

pwd

# env

NUM_NODES=$(echo $VC_WORKER_HOSTS | tr ',' '\n' | wc -l)
# NUM_NODES=1
HEAD_HOST_FULL=${VC_WORKER_HOSTS%%,*}
HEAD_HOST_SHORT=${HEAD_HOST_FULL%%.*} 

MY_IP=$(hostname -I | awk '{print $1}')

echo "Me:       $(hostname)"

# Environment Variables
export RAY_DEDUP_LOGS=0 CUDA_DEVICE_MAX_CONNECTIONS=1 TORCH_COMPILE_DISABLE=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"eth0"}

echo "Me:       $(hostname)"
echo "Head Short: $HEAD_HOST_SHORT"
echo "Head Full:  $HEAD_HOST_FULL"

# worker
if [[ "$(hostname)" != "$HEAD_HOST_SHORT" ]]; then
    echo "=== I am a WORKER. Connecting to $HEAD_HOST_FULL ==="

    # WAITING LOGIC
    echo "Waiting for Head Node ($HEAD_HOST_FULL:6379)..."
    while ! timeout 1 bash -c "echo > /dev/tcp/$HEAD_HOST_FULL/6379" 2>/dev/null; do
        echo "Head node not ready yet... sleeping 5s"
        sleep 5
    done

    echo "Head node is UP! Connecting now..."
    
    exec ray start --address="$HEAD_HOST_FULL:6379" --node-ip-address="$MY_IP" \
        --num-gpus=8 --disable-usage-stats --block
fi

# head
echo "=== I am the HEAD NODE. Starting cluster... ==="

ray start --head --node-ip-address="$MY_IP" --port=6379 \
    --num-gpus=8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Wait for all workers to join
echo "Waiting for $NUM_NODES nodes to join..."
while true; do
    ACTIVE_NODES=$(ray status 2>/dev/null | grep -c " 1 node_")
    
    echo "Nodes Active: $ACTIVE_NODES / $NUM_NODES"
    
    if [[ "$ACTIVE_NODES" -ge "$NUM_NODES" ]]; then
        echo "Cluster Full! All $NUM_NODES nodes are ready."
        break
    fi
    sleep 5
done


cd /shared/dev/scottcyc/workspace/slime

pwd

source /shared/dev/scottcyc/workspace/slime/scripts/models/qwen3-235B-A22B.sh

MODEL_DIR="/shared/dev/scottcyc/checkpoints/Qwen"
DATA_DIR="/shared/dev/scottcyc/data"
PROMPT_DATA="/shared/dev/scottcyc/data/openthoughts3_math_10k_prompts_slime_v2.jsonl"
# PROMPT_DATA="/shared/dev/xth/data/rlvr/math_ds100_43k_slime_passrate_no_reward.jsonl"


CKPT_ARGS=(
    --hf-checkpoint ${MODEL_DIR}/Qwen3-235B-A22B-Thinking-2507
    # --hf-checkpoint ${MODEL_DIR}/GLM-4.5-Air-Base-sft-ift21-rl_239
    # --ref-load ${MODEL_DIR}/GLM-4.5-Air-Base_torch_dist
    # --load ${MODEL_DIR}/GLM-4.5-Air-Base-sft-ift21_torch_dist/
    # --save ${MODEL_DIR}/GLM-4.5-Air-Base-sft-ift21_slime_rl_4/
    # --save-interval 500
)


ROLLOUT_ARGS=(
   --prompt-data $PROMPT_DATA
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rm-type deepscaler
   --num-rollout 20
   --start-rollout-id 0
   --rollout-batch-size 512
   --n-samples-per-prompt 16
   --rollout-max-response-len 32768
   --rollout-temperature 0.9

   --num-steps-per-rollout 4
   --balance-data
   --rollout-stop-token-ids 151645 151643

   --dump-details $DATA_DIR/Qwen/qwen235b_rollout_cluster-3
   --debug-rollout-only
)
#    --metadata-key extra_info



# WANDB_API_KEY="3216efad078e7c21e4b785ef4db3fc2fe2b04996"
WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev-rollout
   --wandb-group qwen-235b-thinking-test-rollout
   --wandb-key ${WANDB_API_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.8
   # --sglang-enable-dp-attention
   # --sglang-dp-size 2

)






# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
\"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\"
}
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --rollout-num-gpus 128 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}


sleep 100