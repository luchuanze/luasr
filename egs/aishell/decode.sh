#!/bin/bash


dir=exp/transducer13
result_file=result117
cer_file=cer117
decode_checkpoint=117.pt

avg_checkpoint=true
avg_num=5

rm $dir/$result_file
touch $dir/$result_file


if [ $avg_checkpoint == true ]; then
	decode_checkpoint=avg_${avg_num}.pt
	echo "do model average and final checkpoint is $decode_checkpoint"
	python3 ../../nets/bin/average_model.py \
          --dst_model $dir/$decode_checkpoint \
          --src_path $dir  \
          --num ${avg_num} \
          --val_best
        
        result_file=avg_result
        cer_file=avg_cer
fi
	

python3 ../../nets/bin/decode.py  \
        --gpu 0 \
	--config $dir/train.yaml \
	--checkpoint $dir/$decode_checkpoint \
	--data_type raw \
	--symbol_table /asr_lu/wenet-main/examples/aishell/s0/data/dict/lang_char2.txt \
	--batch_size 1 \
	--test_data /asr_lu/wenet-main/examples/aishell/s0/data/test/data.list \
	--beam_size 1 \
	--result_file $dir/$result_file


python3 ../../nets/bin/compute-wer.py --char=1 --v-1 /asr_lu/wenet-main/examples/aishell/s0/data/test/text \
        $dir/$result_file > $dir/$cer_file

