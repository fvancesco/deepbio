#!/bin/bash
#$ -N TorchMul4
#$ -cwd
#$ -e $HOME/logs/$JOB_NAME-$JOB_ID.err
#$ -o $HOME/logs/$JOB_NAME-$JOB_ID.out
#$ -l h=node04
#$ -l gpu=1
#$ -q default.q

module load cuda/7.5

########
cd /homedtic/fbarbieri/
. /homedtic/fbarbieri/torch/install/bin/torch-activate

export LD_LIBRARY_PATH="/soft/openblas/openblas-0.2.18/lib:/homedtic/fbarbieri/libraries/hdf5-1.8.17/lib:/homedtic/fbarbieri/libraries/cudnn/cuda/lib64:$LD_LIBRARY_PATH"
export CUDA_TOOLKIT_ROOT_DIR="/soft/cuda/cudnn/cuda/lib64:/soft/cuda/cudnn/cuda/include"
export PROTOBUF_LIBRARY="/usr/lib"
export PROTOBUF_INCLUDE_DIR="/usr/include/google/protobuf"
########

cd /homedtic/fbarbieri/git/deepbio
th train1.lua -learning_rate 0.00001 -hidden_size 50000 > ~/deepbio/logs/4_deepbio_$(date +%Y_%m_%d_%H_%M_%S) &
sleep 1
th train1.lua -learning_rate 0.00001 -hidden_size 5000 > ~/deepbio/logs/4_deepbio_$(date +%Y_%m_%d_%H_%M_%S) &
sleep 1
th train1.lua -learning_rate 0.1 -hidden_size 50000 > ~/deepbio/logs/4_deepbio_$(date +%Y_%m_%d_%H_%M_%S) &
#th train_hpc.lua -val_size 2000 -hidden_size 3000 -print_start 15 -save_output 990 > ~/mmtwitter_old/logs/relu_pre_$(date +%Y_%m_%d_%H_%M_%S) &
#sleep 1
#th train_hpc.lua -val_size 2000 -hidden_size 9000 -print_start 15 -save_output 990 > ~/mmtwitter_old/logs/relu_pre_$(date +%Y_%m_%d_%H_%M_%S) &


wait
echo "Done with 04"

