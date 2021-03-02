python run_lm_hstuning.py \
  --bert_model bert-base-uncased \
  --do_lower_case \
  --do_train \
  --train_file data/all_data.txt \
  --output_dir models \
  --num_train_epochs 5.0 \
  --learning_rate 3e-5 \
  --train_batch_size 16 \
  --max_seq_length 128 \
