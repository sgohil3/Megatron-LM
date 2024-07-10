#!/bin/bash
DIR_PROJECT=""
DIR_MEGATRON_LM="${DIR_PROJECT}/Megatron-LM"
DIR_TRANSFORMER_ENGINE="${DIR_PROJECT}/TransformerEngine"
DIR_APEX="${DIR_PROJECT}/apex"
DIR_GPT2_DATASET="${DIR_PROJECT}/dataset_gpt2"
DIR_MINICONDA="/root/miniconda3"
# Let's begin
# Get Megatron-LM project and prep conda environment
mkdir -p ${DIR_MEGATRON_LM} && cd ${DIR_MEGATRON_LM}
# wget https://github.com/PAakash/Megatron-LM/archive/refs/tags/profile.tar.gz -O ${DIR_MEGATRON-LM}/Megatron-LM.tar.gz
cp /tmp/Megatron-LM.tar.gz ${DIR_MEGATRON_LM}/
tar xzf ${DIR_MEGATRON_LM}/Megatron-LM.tar.gz -C ${DIR_MEGATRON_LM} --strip-components 1
rm -r ${DIR_MEGATRON_LM}/Megatron-LM.tar.gz
${DIR_MINICONDA}/bin/conda activate Megatron-LM_pyEnv
# Get Transformer engine and install for conda environment
mkdir -p ${DIR_TRANSFORMER_ENGINE} && cd ${DIR_TRANSFORMER_ENGINE}
wget https://github.com/NVIDIA/TransformerEngine/archive/refs/tags/v1.1.tar.gz -O ${DIR_TRANSFORMER_ENGINE}/TransformerEngine.tar.gz
tar xzf ${DIR_TRANSFORMER_ENGINE}/TransformerEngine.tar.gz -C ${DIR_TRANSFORMER_ENGINE} --strip-components 1
rm -r ${DIR_TRANSFORMER_ENGINE}/TransformerEngine.tar.gz
MAX_JOBS=1 pip install --no-build-isolation ./
# Get apex and install for conda environment
mkdir -p ${DIR_APEX} && cd ${DIR_APEX}
wget https://github.com/NVIDIA/apex/archive/refs/tags/23.08.tar.gz -O ${DIR_APEX}/apex.tar.gz
tar xzf ${DIR_APEX}/apex.tar.gz -C ${DIR_APEX} --strip-components 1
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
rm -r ${DIR_APEX}/apex.tar.gz
# Prepare data set
mkdir -p ${DIR_GPT2_DATASET} && cd ${DIR_GPT2_DATASET}
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
aws s3 cp s3://neuron-s3/training_datasets/gpt/wikipedia/my-gpt2_text_document.bin .  --no-sign-request
aws s3 cp s3://neuron-s3/training_datasets/gpt/wikipedia/my-gpt2_text_document.idx .  --no-sign-request
aws s3 cp s3://neuron-s3/training_datasets/gpt/wikipedia/license.txt .  --no-sign-request

