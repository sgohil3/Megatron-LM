# #!/bin/bash
# # To run this script 
# # ./profilingScript.sh DP NumberOfGpus
# # ./profilingScript.sh DP_TP NumberOfGpus TP_Degree
# # ./profilingScript.sh DP_PP NumberOfGpus PP_Degree
# # ./profilingScript.sh DP_TP_PP NumberOfGpus TP_Degree PP_Degree

# # Runs the "345M" parameter model
PRJ_DIR="prj_directory_path"
#source ${PRJ_DIR}/idlab_scripts/prep_env.sh
cd ${PRJ_DIR}

PARALLEL_STRAT=$1
	PUS_PER_NODE=$2  

DEGREE_DP=1
DEGREE_PP=1
DEGREE_TP=1

case $PARALLEL_STRAT in
	DP)
		echo "Data parallel"
		GPT_ARGS_DIST=(
		# data parallel degree is all GPUs
		)
		echo DP degree $GPUS_PER_NODE
		DEGREE_DP=$GPUS_PER_NODE
		;;

	DP_TP)
		echo "Data and tensor parallel"
		DEGREE_TP=$3
		GPT_ARGS_DIST=(
		--tensor-model-parallel-size $DEGREE_TP
		)
		echo DP degree $(($GPUS_PER_NODE/$DEGREE_TP))
		echo TP degree $DEGREE_TP
		DEGREE_DP=$(($GPUS_PER_NODE/$DEGREE_TP))
		;;
	
	DP_PP)
		echo "Data and pipeline parallel"
		DEGREE_PP=$3
		GPT_ARGS_DIST=(
		--pipeline-model-parallel-size $DEGREE_PP
		)
		echo DP degree $(($GPUS_PER_NODE/$DEGREE_PP))
		echo PP degree $DEGREE_PP
		DEGREE_DP=$(($GPUS_PER_NODE/$DEGREE_PP))
		;;
	
	DP_TP_PP)
		echo "Data, tensor and pipeline parallel"
		DEGREE_TP=$3
		DEGREE_PP=$4
		GPT_ARGS_DIST=(
		--tensor-model-parallel-size $DEGREE_TP
		--pipeline-model-parallel-size $DEGREE_PP
		)
		echo DP degree $(($GPUS_PER_NODE/($DEGREE_PP*$DEGREE_TP)))
		echo TP degree $DEGREE_TP
		echo PP degree $DEGREE_PP
		DEGREE_DP=$(($GPUS_PER_NODE/($DEGREE_PP*$DEGREE_TP)))
		;;

	DP_TP_SP)
		echo "Data, tensor and sequence parallel"
		DEGREE_TP=$3
		GPT_ARGS_DIST=(
		--tensor-model-parallel-size $DEGREE_TP
		--sequence-parallel
		)
		echo DP degree $(($GPUS_PER_NODE/$DEGREE_TP))
		echo TP degree $DEGREE_TP
		DEGREE_DP=$(($GPUS_PER_NODE/$DEGREE_TP))
		;;
	
	DP_TP_PP_SP)
		echo "Data, tensor, pipeline and sequence parallel"
		DEGREE_TP=$3
		DEGREE_PP=$4
		GPT_ARGS_DIST=(
		--tensor-model-parallel-size $DEGREE_TP
		--pipeline-model-parallel-size $DEGREE_PP
		--sequence-parallel
		)
		echo DP degree $(($GPUS_PER_NODE/($DEGREE_PP*$DEGREE_TP)))
		echo TP degree $DEGREE_TP
		echo PP degree $DEGREE_PP
		DEGREE_DP=$(($GPUS_PER_NODE/($DEGREE_PP*$DEGREE_TP)))
		;;

	*)
		echo "Invalid parallelism"
		return 1
		;;
esac

echo Parameters to Megatron Model training ${GPT_ARGS_DIST[@]}

export CUDA_DEVICE_MAX_CONNECTIONS=1

CHECKPOINT_PATH=${PRJ_DIR}/checkpoint
DATASET_PATH=${PRJ_DIR}/dataset_gpt2
MEGATRON_PATH="${PRJ_DIR}/Megatron-LM"

cDateTime=$( date '+%F_%H%M%S' )
dir_name="DP_${DEGREE_DP}_1_TP_${DEGREE_TP}_1_PP_${DEGREE_PP}_1"
profileOutputs="$MEGATRON_PATH/profile_results/${dir_name}/${cDateTime}"

mkdir -p $MEGATRON_PATH/profile_results/${dir_name}

cd ${MEGATRON_PATH}
#gitst=$( git log --pretty=format:'%cd: %H' -n 1 )
cd ${PRJ_DIR}
echo $dir_name > ${profileOutputs}_profile_log.txt
echo $profileOutputs >> ${profileOutputs}_profile_log.txt
#echo $gitst >> ${profileOutputs}_profile_log.txt

echo " " >> ${profileOutputs}_profile_log.txt
echo " " >> ${profileOutputs}_profile_log.txt
# AP: CLeaning up previous checkpoint for fresh training
rm -r $CHECKPOINT_PATH/*

# Change for multinode config
MASTER_ADDR=localhost #127.0.0.1 #
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
--eval-iters 2 #change this for batch size
--save $CHECKPOINT_PATH
--load ${CHECKPOINT_PATH}/"empty"
--profile
--profile-step-start=9 # use this to control which iteration to profile : use start iteration -> (end iteration + 1)
--profile-step-end=10 
)



PROFILER_ARGS_NCU_SINGLE=(
# ncu arguements for a single GPU
--set full #profile all sections
--verbose #add more data to profiler output
--section WorkloadDistribution
# --section MemoryWorkloadAnalysis
--section MemoryWorkloadAnalysis_Chart 
--section MemoryWorkloadAnalysis_Tables
--section PmSampling
--replay-mode application # use application replay mode (range modes not working need to investigate)
--app-replay-mode strict #all kernels must match, otherwise throw error
--cache-control none
--nvtx --nvtx-include "iteration9/"
# --range-filter :1: #use the first cudaProfilerStart/Stop range for profiling
# --app-replay-match name # Kernels are matched in the following order: 1. (mangled) name, 2. order of execution
-o ${profileOutputs}
)

PROFILER_ARGS_NCU_MULTI=(
# ncu arguements for multiple GPUS
--metrics launch__grid_size
--target-processes all
--replay-mode app-range #application #range
--cache-control none
--nvtx --nvtx-include "iteration9/"
-o ${profileOutputs}
# --range-filter :1: #use the first cudaProfilerStart/Stop range for profiling
)

PROFILER_ARGS_NSYS=(
# # #Uncomment for nsys
-t cuda,nvtx,cudnn,cublas,mpi,ucx 
-s cpu 
--gpu-metrics-device all 
--stats=true 
--capture-range=cudaProfilerApi 
--capture-range-end=stop 
--cudabacktrace all 
-x true 
--cuda-memory-usage true 
--force-overwrite true 
--nic-metrics true
--export hdf,json
-o ${profileOutputs} 
)

echo "Profiler arguments: "
echo ${PROFILER_ARGS[@]}
echo "Distributed arguments"
echo ${DISTRIBUTED_ARGS[@]}
echo "GPT model arguments"
echo ${GPT_MODEL_ARGS[@]}
echo "Training arguments"
echo ${TRAINING_ARGS[@]}
echo "Parallel arguments"
echo ${GPT_ARGS_DIST[@]}
echo "Dataset arguments"
echo ${DATA_ARGS[@]}
echo "Output arguments"
echo ${OUTPUT_ARGS[@]}

# nsys profile ${PROFILER_ARGS_NSYS[@]} \
#     torchrun ${DISTRIBUTED_ARGS[@]} ${MEGATRON_PATH}/pretrain_gpt.py \
#     ${GPT_MODEL_ARGS[@]} \
#     ${TRAINING_ARGS[@]} \
#     ${GPT_ARGS_DIST[@]} \
#     ${DATA_ARGS[@]} \
#     ${OUTPUT_ARGS[@]} #>> ${profileOutputs}_profile_log.txt

ncu ${PROFILER_ARGS_NCU_MULTI[@]} \
    torchrun ${DISTRIBUTED_ARGS[@]} ${MEGATRON_PATH}/pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${GPT_ARGS_DIST[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} >> ${profileOutputs}_profile_log.txt