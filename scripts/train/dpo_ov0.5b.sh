export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=1

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

# 设定gpu
# export CUDA_VISIBLE_DEVICES=3

# 似乎有效
unset NCCL_SOCKET_IFNAME


# DPO Stage
PROMPT_VERSION="qwen_1_5"
# SFT_MODEL="lmms-lab/llava-onevision-qwen2-0.5b-ov"
SFT_MODEL="/home/yilin/RL-MLLM/MM-RLHF/output/DPO/llava-onevision-qwen2-0.5b-ov_mmrlhf-w0.1-beta0.1-epoch1/"
EPOCH=2
beta=0.1
ls_factor_weight=0.1
DPO_RUN_NAME="llava-onevision-qwen2-0.5b-ov_mmrlhf-w${ls_factor_weight}-beta${beta}-epoch${EPOCH}"
DPO_CLEAN_NAME="${DPO_RUN_NAME##*/}"
OUTPUT_DIR="output/DPO/${DPO_CLEAN_NAME}"
# DATA_PATH="/home/yilin/MM-RLHF/MM-RLHF/dpo_pairs.jsonl"
DATA_PATH="/home/yilin/MM-RLHF/output/ref-data-0.5b.jsonl"
IMAGE_FOLDER="/home/yilin/MM-RLHF-Data/" 
VIDEO_FOLDER="/home/yilin/MM-RLHF-Data/"





echo $DPO_RUN_NAME
# --deepspeed scripts/zero2.json \

# python -m debugpy --connect 5679 $(which torchrun) --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_dpo.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path=${SFT_MODEL} \
    --dpo_alpha 1.0 --beta 0.1 --gamma 0.25 \
    --ls_factor_weight $ls_factor_weight \
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
    --gradient_accumulation_steps 12 \
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
    --dataloader_pin_memory True \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --attn_implementation sdpa \
    --seed 0 \
    --data_seed 0 \
    --is_profiler_output False \
    --profiler_repeat 1 \
    --profiler_active 25 \
    --is_key_avg_output False \
