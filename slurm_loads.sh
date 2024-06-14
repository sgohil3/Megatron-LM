# Project paths
export PRJ_PATH= #project path
# export CONDA_ENV_PATH=Megatron-LM_pyEnv
export CONDA_ENV_PATH=${PRJ_PATH}/pyEnv

export MOD_PATH= #path to mods

# Exports for gpgpu-sim
#export CUDNN_PATH=${MOD_PATH}/cuDNN/8.2.1.32-CUDA-11.3.1
#export NCCL_PATH=${MOD_PATH}/NCCL/2.10.3-GCCcore-10.3.0-CUDA-11.3.1

# Exports for pytorch-gpgpu-sim
#export CUDNN_INCLUDE_DIR=${MOD_PATH}/cuDNN/8.2.1.32-CUDA-11.3.1/bin
#export CUDNN_LIBRARY=${MOD_PATH}/cuDNN/8.2.1.32-CUDA-11.3.1/lib

# Workload paths


# Setting up modules
module purge

export MODULEPATH=""


module load lang/Miniconda3/4.12

module load compiler/GCC/10.3.0
module load devel/make/4.3-GCCcore-10.3.0
# Load version for GCC 10.3.0
module load devel/CMake/3.20.1-GCCcore-10.3.0
module load devel/makedepend/1.0.6-GCCcore-10.3.0
module load lang/Bison/3.7.6-GCCcore-10.3.0
module load lang/flex/2.6.4-GCCcore-10.3.0
module load lib/zlib/1.2.11-GCCcore-10.3.0

module load devel/Doxygen/1.9.1-GCCcore-10.3.0
module load vis/Graphviz/2.47.2-GCCcore-10.3.0

module load lib/libpng/1.6.37-GCCcore-10.3.0

module load lib/NCCL/2.10.3-GCCcore-10.3.0-CUDA-11.3.1
module load system/CUDA/11.8.0
module load numlib/cuDNN/8.7.0.84-CUDA-11.8.0

module load mpi/OpenMPI/4.1.1-GCC-10.3.0
module load lib/libtool/2.4.6-GCCcore-10.3.0 



echo "Module loaded..."


export CUDA_INSTALL_PATH=${CUDA_PATH}

if [ -d ${CONDA_ENV_PATH} ] 
then
    conda activate ${CONDA_ENV_PATH}
else
    echo "Error: Conda environment does not exists."
    conda create --prefix ${CONDA_ENV_PATH} python=3.8
    conda activate ${CONDA_ENV_PATH}
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
fi

echo "Conda environment setup done."

export USE_SYSTEM_NCCL=1
