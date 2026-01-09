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




# cd /shared/dev/scottcyc/workspace/slime

pwd


MODEL_DIR="/shared/dev/scottcyc/checkpoints/Qwen"

source /shared/dev/scottcyc/workspace/slime/scripts/models/qwen3-8B.sh


CKPT_ARGS=(
    --hf-checkpoint ${MODEL_DIR}/Qwen3-8B-Base
    --ref-load ${MODEL_DIR}/Qwen3-8B-Base_torch_dist
    --load ${MODEL_DIR}/Qwen3-8B-Base_sft_ring_16_r1/
    --save ${MODEL_DIR}/Qwen3-8B-Base_sft_ring_16_r1/
    --save-interval 500
)

SFT_ARGS=(
    --rollout-function-path slime.rollout.sft_rollout.generate_rollout
    --prompt-data /shared/dev/scottcyc/data/Ring-lite-sft-data.shuf.160k.jsonl
    --input-key messages
    --loss-mask-type qwen3
    --rollout-shuffle
    --num-epoch 1
    --rollout-batch-size 128
    --global-batch-size 128
    --n-samples-per-prompt 1

    --loss-type sft_loss
    --calculate-per-token-loss
    --disable-compute-advantages-and-returns
    --debug-train-only
)

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    # --micro-batch-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 32768
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-5
    --lr-decay-style cosine
    --min-lr 1e-6
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95

    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)

# OPTIMIZER_ARGS=(
#     --optimizer adam
#     --lr 1e-5
#     --lr-warmup-iters 100
#     --lr-decay-style cosine
#     --min-lr 5e-6
#     --weight-decay 0.1
#     --adam-beta1 0.9
#     --adam-beta2 0.95

#     --optimizer-cpu-offload
#     --overlap-cpu-optimizer-d2h-h2d
#     --use-precision-aware-optimizer
# )

WANDB_ARGS=(
    --use-wandb
    --wandb-project slime-dev-qwen
    --wandb-group qwen-8B-sft
    --wandb-key ${WANDB_API_KEY}
)

MISC_ARGS=(
    # default dropout in megatron is 0.1
    --attention-dropout 0.0
    --hidden-dropout 0.0
    # should be good for model performance
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    # need to comment this when using model with MLA
    --attention-backend flash
    --make-vocab-size-divisible-by 1
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
    --actor-num-nodes $NUM_NODES \
    --actor-num-gpus-per-node 8 \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${SFT_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${WANDB_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${EVAL_ARGS[@]} \
    ${MISC_ARGS[@]}
