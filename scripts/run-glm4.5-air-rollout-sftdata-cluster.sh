#!/bin/bash
set -euo pipefail
set -x

echo "=== Starting unified Ray worker node ==="
echo "Hostname: $(hostname)"
echo "IP: $(hostname -I | awk '{print $1}')"
env | grep -E "(VC_|HOSTNAME)" | sort


pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
# pkill -9 python
sleep 3
pkill -9 ray
# pkill -9 python

# Environment setup
export EXPECTED_NODES=${EXPECTED_NODES:-1}
# export EXPECTED_NODES=8
export RAY_DEDUP_LOGS=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_COMPILE_DISABLE=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"eth0"}
export MLP_SOCKET_IFNAME=${MLP_SOCKET_IFNAME:-"eth0"}

RAY_PORT_NUMBER=6379
MY_IP=$(hostname -I | awk '{print $1}')

# Function to check if the number of connected nodes equals the expected number
wait_for_nodes() {
    local expected_nodes=$1
    local timeout=$2
    local interval=10
    local elapsed=0

    echo "Waiting for $expected_nodes nodes to join the cluster..."

    while true; do
        connected_nodes=$(ray status | awk '/Active:/,/Pending:/' | grep '^ 1' | wc -l || echo "0")
        echo "There are $connected_nodes nodes connected to the cluster"

        if [[ $connected_nodes == $expected_nodes ]]; then
            echo "Expected $connected_nodes nodes are connected to the cluster. Ray cluster is ready"
            echo "Fetching Ray status."
            ray status
            break
        fi

        sleep $interval
        elapsed=$((elapsed + interval))
        if [[ $elapsed -ge $timeout ]]; then
            echo "Timeout(${timeout}) waiting for nodes to join the cluster. Exiting."
            exit 1
        fi
    done
}

# Start Ray cluster first - determine if this is worker-0 (which will be the head node)
if [[ "$(hostname)" == *"-worker-0" ]]; then
    echo "=== I am worker-0, starting as Ray HEAD node ==="
    
    # Start Ray head node
    echo "Starting Ray head node..."
    ray start --head --node-ip-address $MY_IP --port=${RAY_PORT_NUMBER} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
    
    # Wait for all worker nodes to connect
    wait_for_nodes $EXPECTED_NODES 3600  # 1 hour timeout

else
    echo "=== I am a worker node, connecting to worker-0 ==="
    
    # Wait a bit for the head node to start
    sleep 10
    
    # Get the head node address using Volcano service discovery
    HEAD_NODE_HOSTNAME=""
    if [ -n "${VC_WORKER_HOSTS:-}" ]; then
        # Extract worker-0 hostname from the comma-separated list
        HEAD_NODE_HOSTNAME=$(echo "${VC_WORKER_HOSTS}" | cut -d',' -f1)
        echo "Head node hostname from VC_WORKER_HOSTS: ${HEAD_NODE_HOSTNAME}"
    fi
    
    if [ -z "$HEAD_NODE_HOSTNAME" ]; then
        echo "Could not determine head node hostname, exiting"
        exit 1
    fi
    
    # Wait for head node to be ready
    echo "Waiting for head node at ${HEAD_NODE_HOSTNAME}:${RAY_PORT_NUMBER} to be ready..."
    for i in {1..60}; do
        if timeout 5 bash -c "echo > /dev/tcp/${HEAD_NODE_HOSTNAME}/${RAY_PORT_NUMBER}" 2>/dev/null; then
            echo "Head node is ready!"
            break
        fi
        echo "Attempt $i: Head node not ready yet, waiting..."
        sleep 10
    done
    
    # Connect to Ray cluster
    echo "Connecting to Ray cluster at ${HEAD_NODE_HOSTNAME}:${RAY_PORT_NUMBER}"
    ray start --address=${HEAD_NODE_HOSTNAME}:${RAY_PORT_NUMBER} --num-gpus 8 --node-ip-address ${MY_IP} --disable-usage-stats
    
    # Keep the worker alive
    echo "Ray worker connected successfully. Keeping alive..."
    while true; do
        echo "Worker heartbeat at $(date)"
        sleep 300  # 5 minutes
    done
fi



# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"



# cd /shared/dev/scottcyc/workspace/slime

pwd

source /shared/dev/scottcyc/workspace/slime/scripts/models/glm4.5-106B-A12B.sh

MODEL_DIR="/shared/dev/scottcyc/checkpoints/GLM"
DATA_DIR="/shared/dev/scottcyc/data"
PROMPT_DATA="/shared/dev/scottcyc/data/openthoughts3_math_10k_prompts_slime.jsonl"
# PROMPT_DATA="/shared/dev/xth/data/rlvr/math_ds100_43k_slime_passrate_no_reward.jsonl"


CKPT_ARGS=(
    --hf-checkpoint ${MODEL_DIR}/GLM-4.5-Air
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
   --num-rollout 1250
   --start-rollout-id 0
   --rollout-batch-size 8
   --n-samples-per-prompt 16
   --rollout-max-response-len 32768
   --rollout-temperature 0.9

   --num-steps-per-rollout 4
   --balance-data
   --rollout-stop-token-ids 151329 151336 151338

   --dump-details $DATA_DIR/GLM/glm106b_rollout_cluster-0
   --debug-rollout-only
)
#    --metadata-key extra_info



# WANDB_API_KEY="3216efad078e7c21e4b785ef4db3fc2fe2b04996"
WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev-rollout
   --wandb-group glm-106b-test-rollout
   --wandb-key ${WANDB_API_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.85
   # --sglang-enable-dp-attention
   # --sglang-dp-size 2

)






# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --rollout-num-gpus 64 \
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