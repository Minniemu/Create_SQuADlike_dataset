# Create_SQuADlike_dataset


### Steps
1. 讀取 BERT_QG 產生的 Question-Answer pair(eval_beam_size_3.json)




### Evaluate 
1. We are trying to use hugging face question-answering model.(we use v4.0.0 here)
2. Instruction : 
```
CUDA_VISIBLE_DEVICES=0 python run_squad.py --model_type bert --model_name_or_path bert-base-uncased --do_eval --do_lower_case --predict_file squad_v1.1.json --per_gpu_train_batch_size 6 --learning_rate 3e-5 --max_seq_length 384 --doc_stride 128 --output_dir /tmp/debug_squad
```
**To prevent memory overflow of 1060,we make the batch size 6(you can change it to 12 if you are running on a better graphic card).**