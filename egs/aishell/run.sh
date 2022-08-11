#!/bin/bash

python3 ../../nets/bin/train2.py  \
	--config conf/aishell.yaml \
	--data_type raw \
	--symbol_table /asr_lu/wenet-main/examples/aishell/s0/data/dict/lang_char2.txt \
     	--train_data /asr_lu/wenet-main/examples/aishell/s0/data/train/data.list \
     	--cv_data /asr_lu/wenet-main/examples/aishell/s0/data/dev/data.list \
     	--cmvn /asr_lu/wenet-main/examples/aishell/s0/exp/conformer_unified/global_cmvn \
 	--num_workers 2 \
	--model_dir exp/transducer
