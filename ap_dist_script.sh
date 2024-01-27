#!/bin/bash

# Runs the "345M" parameter model

cd /project_antwerp/gpu_lite
source ./idlab_scripts/prep_env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1

CHECKPOINT_PATH=/project_antwerp/gpu_lite/checkpoint
DATASET_PATH=/project_antwerp/gpu_lite/dataset_gpt2
MEGATRON_PATH=/project_antrwerp/gpu_lite/prj/Megatron-LM
# AP: CLeaning up previous checkpoint for fresh training
rm -r $CHECKPOINT_PATH/*

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=$CHECKPOINT_PATH
VOCAB_FILE=$DATASET_PATH/gpt2-vocab.json
MERGE_FILE=$DATASET_PATH/gpt2-merges.txt
DATA_PATH=$DATASET_PATH/my-gpt2_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NNODES 
    --node_rank $NODE_RANK 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
    )

GPT_MODEL_ARGS=(
    --num-layers 24 
    --hidden-size 1024 
    --num-attention-heads 16 
    --seq-length 1024
    --max-position-embeddings 1024 
    )

TRAINING_ARGS=(
    --micro-batch-size 4 
    --global-batch-size 16 
    --train-iters 20
    --weight-decay 1e-2
    --clip-grad 1.0 
    --fp16
    --lr 0.00015 
    --lr-decay-style cosine 
    --min-lr 1.0e-5
    --lr-warmup-fraction .01 
    --lr-decay-iters 320000 
)

# Data parallel
GPT_ARGS_DP=(
#    --use-cpu-initialization    
)

# DP + TP
GPT_ARGS_DP_TP=(
    --tensor-model-parallel-size 2
    )
# DP + PP
GPT_ARGS_DP_PP=(
    --pipeline-model-parallel-size 2
    )

# DP + TP + PP
GPT_ARGS_DP_TP_PP=(
    --tensor-model-parallel-size 2 
    --pipeline-model-parallel-size 2 
    )

# DP + TP + SP
GPT_ARGS_DP_TP_SP=(
    --tensor-model-parallel-size 2 
    --sequence-parallel
    )

# DP + TP + SP + PP
GPT_ARGS_DP_TP_PP_SP=(
    --tensor-model-parallel-size 2 
    --pipeline-model-parallel-size 2 
    --sequence-parallel
    )

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
#    --split 949,50,1
    )

OUTPUT_ARGS=(
    --log-interval 100 
    --save-interval 10000 
    --eval-interval 1000 
    --eval-iters 10
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --profile
    --profile-step-start=0

    )

PROFILER_ARGS=(
    -t cuda,nvtx,cudnn,cublas 
    -s cpu 
    --gpu-metrics-device all 
    --stats=true 
    --capture-range=cudaProfilerApi 
    --capture-range-end=stop 
    --cudabacktrace all 
    -x true 
    --cuda-memory-usage true 
    --force-overwrite true 
    -o testProfile 
    )

nsys profile ${PROFILER_ARGS[@]} \
        torchrun ${DISTRIBUTED_ARGS[@]} ${MEGATRON_PATH}pretrain_gpt.py \
        ${GPT_MODEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${GPT_ARGS_DP[@]} \
        ${DATA_ARGS[@]} \
        ${OUTPUT_ARGS[@]} > $CHECKPOINT_PATH/profile_log.txt


