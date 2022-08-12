#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1"

dir=exp/transducer
INIT_FILE=$dir/ddp_init
init_method=file://$(readlink -f $INIT_FILE)
  # Use "nccl" if it works, otherwise use "gloo"
dist_backend="gloo"

num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

for ((i = 0; i < $num_gpus; ++i));do
{
  rank=$i
  python3 ../../nets/bin/train2.py  \
	--config conf/aishell.yaml \
	--data_type raw \
	--symbol_table /asr_lu/wenet-main/examples/aishell/s0/data/dict/lang_char2.txt \
     	--train_data /asr_lu/wenet-main/examples/aishell/s0/data/train/data.list \
     	--cv_data /asr_lu/wenet-main/examples/aishell/s0/data/dev/data.list \
     	--cmvn /asr_lu/wenet-main/examples/aishell/s0/data/train/global_cmvn \
 	--num_workers 2 \
 	--pin_memory \
	--prefetch 1000 \
	--ddp.rank $rank \
 	--ddp.world_size 2 \
 	--ddp.init_method $init_method \
 	--ddp.dist_backend $dist_backend \
	--model_dir exp/transducer
} &
done
wait


