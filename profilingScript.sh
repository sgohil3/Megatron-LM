# #!/bin/bash
# # To run this script 
# # ./profilingScript.sh NCU_OR_NSYS DP NumberOfGpus ITERATION_START
# # ./profilingScript.sh NCU_OR_NSYS DP_TP NumberOfGpus TP_Degree ITERATION_START
# # ./profilingScript.sh NCU_OR_NSYS DP_PP NumberOfGpus PP_Degree ITERATION_START
# # ./profilingScript.sh NCU_OR_NSYS DP_TP_PP NumberOfGpus TP_Degree PP_Degree ITERATION_START
# # example: ./profilingScript.sh NCU_MULTI DP_TP_PP 4 2 2 0

# # Runs the "345M" parameter model
PRJ_DIR="/imec/scratch/dtpatha/gohil01/test2/prj"
cd ${PRJ_DIR}
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

LAUNCH_COUNT=340 #change this number to alter how many kernels are altered per script execution with NCU_MULTI
PROFILE_MODE=$1
PARALLEL_STRAT=$2
GPUS_PER_NODE=$3  
ITERATION_COUNT=$4

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
		DEGREE_TP=$4
		ITERATION_COUNT=$5
		GPT_ARGS_DIST=(
		--tensor-model-parallel-size $DEGREE_TP
		)
		echo DP degree $(($GPUS_PER_NODE/$DEGREE_TP))
		echo TP degree $DEGREE_TP
		DEGREE_DP=$(($GPUS_PER_NODE/$DEGREE_TP))
		;;
	
	DP_PP)
		echo "Data and pipeline parallel"
		DEGREE_PP=$4
		ITERATION_COUNT=$5
		GPT_ARGS_DIST=(
		--pipeline-model-parallel-size $DEGREE_PP
		)
		echo DP degree $(($GPUS_PER_NODE/$DEGREE_PP))
		echo PP degree $DEGREE_PP
		DEGREE_DP=$(($GPUS_PER_NODE/$DEGREE_PP))
		;;
	
	DP_TP_PP)
		echo "Data, tensor and pipeline parallel"
		DEGREE_TP=$4
		DEGREE_PP=$5
		ITERATION_COUNT=$6
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
		DEGREE_TP=$4
		ITERATION_COUNT=$5
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
		DEGREE_TP=$4
		DEGREE_PP=$5
		ITERATION_COUNT=$6
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
MEGATRON_PATH=${PRJ_DIR}/Megatron-LM

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
--profile-step-start=8 # use this to control which iteration to profile : use start iteration -> (end iteration + 1)
--profile-step-end=9
)

PROFILER_ARGS_NCU_SINGLE=(
# ncu arguements for a single GPU
--verbose #add more data to profiler output
--filter-mode per-gpu
--section MemoryWorkloadAnalysis_Tables
--section MemoryWorkloadAnalysis_Chart
--section SpeedOfLight_HierarchicalTensorRooflineChart
--section Nvlink
--section Nvlink_Tables
--section Nvlink_Topology
--metric pcie__write_bytes.sum,pcie__read_bytes.sum
--metric l1tex__m_l1tex2xbar_write_bytes.sum,dram__bytes_write.sum,l1tex__m_xbar2l1tex_read_bytes.sum,dram__bytes_read.sum
# --metric sm__sass_l1tex_m_xbar2l1tex_read_bytes_mem_global_op_ldgsts_cache_bypass.sum,
# --metric gpu__time_duration.sum,pcie__write_bytes.sum,pcie__read_bytes.sum,\
# --metric sm__ops_path_tensor_src_fp16_dst_fp16,sm__ops_path_tensor_src_fp16_dst_fp32
# --metric lts__t_sectors_srcunit_tex_aperture_sysmem_op_read_lookup_miss.sum
# --metric lts__t_sectors_srcunit_tex_aperture_sysmem_op_write_lookup_miss.sum,lts__t_sectors_srcunit_tex_aperture_sysmem_op_red_lookup_miss.sum #,\
# --metric lts__t_sectors_aperture_peer.sum,lts__t_sectors_aperture_peer_op_membar.sum,lts__t_sectors_aperture_peer_op_read.sum,\
# --metric lts__t_sectors_aperture_peer_op_write.sum,lts__t_sectors_srcunit_l1_aperture_peer.sum,lts__t_sectors_srcunit_ltcfabric_aperture_peer.sum,lts__t_sectors_srcunit_tex_aperture_peer.sum,\
# --metric l1tex__m_l1tex2xbar_write_bytes.sum,dram__bytes_write.sum
# --metric l1tex__m_xbar2l1tex_read_bytes.sum 
# --metric dram__bytes_read.sum
# --metric lts__t_sectors_srcunit_tex_aperture_peer_op_read_lookup_miss.sum
# --metric lts__t_sectors_srcunit_tex_aperture_peer_op_red_lookup_miss.sum,lts__t_sectors_srcunit_tex_aperture_peer_op_write_lookup_miss.sum
#use the ones below for A100/H100
# --metric lts__t_sectors_srcunit_ltcfabric.sum 
# --metric sm__sass_l1tex_m_xbar2l1tex_read_bytes_mem_global_op_ldgsts_cache_bypass.sum,gpu__time_duration.sum	 
--app-replay-match name
--replay-mode application # use application replay mode (range doesn't work due to unsupported API calls) (app-range doesn't work; don't know why yet)
--cache-control none
--range-filter :1:
# --nvtx --nvtx-include "NCU/"
-o ${profileOutputs}
)

PROFILER_ARGS_NCU_MULTI=(
# ncu arguements for multiple GPUS
--verbose
--launch-skip `expr $LAUNCH_COUNT \* $ITERATION_COUNT`
--launch-count $LAUNCH_COUNT
--set full
--section MemoryWorkloadAnalysis_Chart
--section SpeedOfLight_HierarchicalTensorRooflineChart
--section Nvlink
--section Nvlink_Tables
--section Nvlink_Topology
--metric pcie__write_bytes.sum,pcie__read_bytes.sum, gpu__time_duration.sum	
--replay-mode kernel
--filter-mode per-gpu
--cache-control none
--range-filter :1:
-o ${profileOutputs}
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
echo "Profiler arguments: "

case $PROFILE_MODE in 
	NSYS)
		echo ${PROFILER_ARGS[@]}
		echo "Starting nsys profiling"
		nsys profile ${PROFILER_ARGS_NSYS[@]} \
			torchrun ${DISTRIBUTED_ARGS[@]} ${MEGATRON_PATH}/pretrain_gpt.py \
			${GPT_MODEL_ARGS[@]} \
			${TRAINING_ARGS[@]} \
			${GPT_ARGS_DIST[@]} \
			${DATA_ARGS[@]} \
			${OUTPUT_ARGS[@]} >> ${profileOutputs}_profile_log.txt
		;;

	NCU_SINGLE)
		echo ${PROFILER_ARGS_NCU_SINGLE[@]}
		echo "starting NCU single gpu profiling"
		ncu ${PROFILER_ARGS_NCU_SINGLE[@]} \
			torchrun ${DISTRIBUTED_ARGS[@]} ${MEGATRON_PATH}/pretrain_gpt.py \
			${GPT_MODEL_ARGS[@]} \
			${TRAINING_ARGS[@]} \
			${GPT_ARGS_DIST[@]} \
			${DATA_ARGS[@]} \
			${OUTPUT_ARGS[@]} >> ${profileOutputs}_profile_log.txt		
		;;

	NCU_MULTI)
		echo ${PROFILER_ARGS_NCU_MULTI[@]}
		echo --launch-skip `expr $LAUNCH_COUNT \* $ITERATION_COUNT`
		echo --launch-count ${LAUNCH_COUNT}
		echo "starting NCU multi gpu profiling"
		ncu ${PROFILER_ARGS_NCU_MULTI[@]} \
			torchrun ${DISTRIBUTED_ARGS[@]} ${MEGATRON_PATH}/pretrain_gpt.py \
			${GPT_MODEL_ARGS[@]} \
			${TRAINING_ARGS[@]} \
			${GPT_ARGS_DIST[@]} \
			${DATA_ARGS[@]} \
			${OUTPUT_ARGS[@]} >> ${profileOutputs}_profile_log.txt
		;;
	*)
esac
