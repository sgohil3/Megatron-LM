# module load lang/Miniconda3/4.12

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

# # module load lib/NCCL/2.10.3-GCCcore-10.3.0-CUDA-11.3.1
# # module load lib/NCCL/2.10.3-GCCcore-11.2.0-CUDA-11.4.1
module load numlib/cuDNN/8.7.0.84-CUDA-11.8.0

module load mpi/OpenMPI/4.1.1-GCC-10.3.0
module load lib/libtool/2.4.6-GCCcore-10.3.0 
module unload system/OpenSSL/1.1 #breaks pip install
module unload lang/Python/3.9.5-GCCcore-10.3.0 #needed for proper anaconda enviro
module load lib/NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.1.1
module load system/CUDA/11.8.0

