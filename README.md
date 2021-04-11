---
tags : Github
---

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


### Problems
* Predict (dev_v1.1.json)
```
Results: {'exact': 59.23368022705771,'f1': 68.35083996742, 'total': 10570, 'HasAns_exact': 59.23368022705771, 'HasAns_f1': 68.35083795996742, 'Has_total': 10570, 'best_exact': 59.23368022705771, 'best_exact_thresh': 0.0, 'best_f1': 68.35085996742, 'best_f1_thresh': 0.0}
```
* With pretrained model
```
Results: {'exact': 0.34058656575212864, 'f1': 7.720321942106521, 'total': 10570, 'HasAns_exact': 0.34058656575212864, 'HasAns_f1': 7.720321942106521, 'HasAns_total': 10570, 'best_exact': 0.34058656575212864, 'best_exact_thresh': 0.0, 'best_f1': 7.720321942106521, 'best_f1_thresh': 0.0}

```
* Predict (my own dataset)
```
Results: {'exact': 59.23368022705771, 'f1': 68.35083795996742, 'total': 10570, 'HasAns_exact': 59.23368022705771, 'HasAns_f1': 68.35083795996742, 'HasAns_total': 10570, 'best_exact': 59.23368022705771, 'best_exact_thresh': 0.0, 'best_f1': 68.35083795996742, 'best_f1_thresh': 0.0}
```