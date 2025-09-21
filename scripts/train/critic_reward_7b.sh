export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
# export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# 手动输入参数
# yilin
NUM_GPUS=2
NNODES=1
PORT=6379
RANK=0
ADDR="127.0.0.1"

# 似乎有效
unset NCCL_SOCKET_IFNAME

# DPO Stage
PROMPT_VERSION="qwen_1_5"
SFT_MODEL="lmms-lab/llava-onevision-qwen2-7b-ov"
EPOCH=1
DPO_RUN_NAME="llava-ov-reward-qwen2-7b-ov_mmrlhf-epoch${EPOCH}"
DPO_CLEAN_NAME="${DPO_RUN_NAME##*/}"
OUTPUT_DIR="./output/reward/${DPO_CLEAN_NAME}"
DATA_PATH="/home/yilin/MM-RLHF/dpo_pairs.jsonl"
IMAGE_FOLDER="/home/yilin/MM-RLHF/" 
VIDEO_FOLDER="/home/yilin/MM-RLHF/"

echo $DPO_RUN_NAME

# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
python -m debugpy --connect 5679 $(which torchrun) --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_dpo.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path=${SFT_MODEL} \
    --version $PROMPT_VERSION \
    --data_path=$DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_spatial_pool_mode bilinear \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $DPO_CLEAN_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 24 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --attn_implementation flash_attention_2 \
    --is_rm True \
    --critic_rewards_weight 1.0 \
    --float_rewards_weight 1.0 \
    --center_rewards_coefficient 0.01 \
