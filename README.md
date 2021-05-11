---
tags : Github
---

# Create_SQuADlike_dataset



Steps
---
1. 讀取 BERT_QG 產生的 Question-Answer pair(eval_beam_size_3.json)，將生成的問題及原始context放到transformers question answering model 中回答，將回答之答案及回答之信心度(score)寫入，並寫入score.json

**Code:**
```python=1
import json
from transformers import pipeline
from tqdm import tqdm 
import torch
result = []
qa = pipeline("question-answering",device = 1)
with open('eval_beam_size_3.json','r') as readfile:
    data = json.load(readfile)
i = 0
for d in tqdm(data):
    try :
        #print("context = ",i)
        temp = {}
        temp['context'] = d['context']
        temp['answers'] = []
        dictionary = {}
        dictionary['text'] = d['answers'][0]['text']
        dictionary['answer_start'] = d['answers'][0]['answer_start']
        temp['answers'].append(dictionary)
        temp['questions'] = []
        temp['answer_list'] = []
        temp['score'] = []
        for q in d['gen_questions']:
            question = q
            context = d['context']
            # Generating an answer to the question in context
            answer = qa(question=question, context=context)
            temp['questions'].append(question)
            temp['answer_list'].append(answer['answer'])
            temp['score'].append(answer['score'])
        result.append(temp)
    except KeyError:
        pass
print(result)

with open('score.json','w') as writefile:
    json.dump(result,writefile)
```
**score.json result:**
```python=1
[
  {
    "context": "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually ",
    "answers": [
      {
        "text": "Norman",
        "answer_start": 4
      }
    ],
    "questions": [
      "what is the name of the normans ?",
      "what is the name for the normans ?",
      "what is the name of the normans that gave normandy its name ?"
    ],
    "answer_list": [
      "Normans",
      "Normans",
      "The Normans"
    ],
    "score": [
      0.6477364897727966,
      0.6960868835449219,
      0.5733014345169067
    ]
  },
  {
    "context": "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually ",
    "answers": [
      {
        "text": "Nourmands",
        "answer_start": 21
      }
    ],
    "questions": [
      "what was the normans ' name ?",
      "what is the normans ' name ?",
      "what normans were the people who gave name to normandy ?"
    ],
    "answer_list": [
      "Normans",
      "Normans",
      "The Normans"
    ],
    "score": [
      0.5407866835594177,
      0.5436832904815674,
      0.6475070118904114
    ]
  }
]
```
2. Filter score.json
* approach : Get the answer in answer_list which is a substring of answer
```python=
import json
from tqdm import tqdm
with open('score.json','r') as read_file:
    data = json.load(read_file)
final = [] #the result to write in result.json
for d in tqdm(data):
    temp = {}
    temp['context'] = d['context'] #Get the context
    temp['answers'] = d['answers'] #Get the answer
    answer = d['answers'][0]['text'] 
    question_len = len(d['questions']) #numbers of questions
    temp['questions'] = [] 
    temp['answer_list'] = []
    #print(answer)
    for i in range(question_len):
        sub = d['answer_list'][i] 
        if sub.find(answer)!=-1: #if we can find the substring of answer in my original answer.
            temp['questions'].append(d['questions'][i]) #than put it into my questions
            temp['answer_list'].append(d['answer_list'][i]) #and also the answer_list
    if len(temp['answer_list']) != 0: #if my answer_list is not empty
        final.append(temp)
with open('result.json','w') as write_file:
    json.dump(final,write_file)
            
```
* Preference of result.json:
```python=1
[
  {
    "context": "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually ",
    "answers": [
      {
        "text": "Norman",
        "answer_start": 4
      }
    ],
    "questions": [
      "what is the name of the normans ?",
      "what is the name for the normans ?",
      "what is the name of the normans that gave normandy its name ?"
    ],
    "answer_list": [
      "Normans",
      "Normans",
      "The Normans"
    ]
  },
  {
    "context": "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually ",
    "answers": [
      {
        "text": "Latin",
        "answer_start": 50
      }
    ],
    "questions": [
      "what language was normanni ?"
    ],
    "answer_list": [
      "Latin"
    ]
  }
]
```
3. Package my own dataset : Because we are going to make my own squad1.1
* Code:
```python=1
import json
from tqdm import tqdm
with open('result.json','r') as read_file:
    data = json.load(read_file)
result = {}
result['data'] = []
i = 1
with open('train-v2.0.json','r') as file:
    squad = json.load(file)
data_array = squad['data']
for d in tqdm(data_array):
    result['data'].append(d)

for d in tqdm(data):  
    temp = {}
    temp['paragraphs'] = []
    temp['title'] = d['answers'][0]['text']
    dictionary = {}
    dictionary['qas'] = []
    context = d['context']
    dictionary['context'] = d['context']
    for q in range(len(d['questions'])):
        t = {}
        t['question'] = d['questions'][q]
        t['id'] = str(i)
        i += 1
        t['answers'] = []
        ans = {}
        ans['text'] = d['answer_list'][q]
        ans['answer_start'] = context.find(ans['text'])
        t['answers'].append(ans)
        dictionary['qas'].append(t)
    temp['paragraphs'].append(dictionary)
    result['data'].append(temp)
with open('squad_v1.1.json','w') as write_file:
    json.dump(result,write_file)  
```
* Squad_v1.1.json:
```python=1

```

Evaluate
---
1. We are trying to use hugging face question-answering model.(we use v4.0.0 here)
2. Instruction : 
* Train
```python=1
CUDA_VISIBLE_DEVICES=0 python run_squad.py --model_type bert --model_name_or_path bert-base-uncased --do_eval --do_train --do_lower_case --train_file squad_v3.4.json --predict_file dev_v1.1.json --gradient_accumulation_steps 1 --per_gpu_train_batch_size 12 --learning_rate 3e-5 --max_seq_length 384 --doc_stride 128 --output_dir /tmp/debug_squad7 --overwrite_cache --save_steps 10000
```
* Predict
```python=1
CUDA_VISIBLE_DEVICES=0 python run_squad.py --model_type bert --model_name_or_path /tmp/debug_squad1 --do_eval --do_lower_case --predict_file squad_v1.2.json --gradient_accumulation_steps 2 --per_gpu_train_batch_size 6 --overwrite_cache --learning_rate 3e-5 --max_seq_length 384 --doc_stride 128 --output_dir /tmp/debug_squad1
```
**To prevent memory overflow of 1060,we make the batch size 6(you can change it to 12 if you are running on a better graphic card).**

3. Result
* Train
```python=1
Results: {'exact': 81.28666035950805, 'f1': 88.39955423776105, 'total': 10570, 'HasAns_exact': 81.28666035950805, 'HasAns_f1': 88.39955423776105, 'HasAns_total': 10570, 'best_exact': 81.28666035950805, 'best_exact_thresh': 0.0, 'best_f1': 88.39955423776105, 'best_f1_thresh': 0.0}
```
* Predict
```python=1
Results: {'exact': 68.31459084800905, 'f1': 77.41966316717496, 'total': 28278, 'HasAns_exact': 68.31459084800905, 'HasAns_f1': 77.41966316717496, 'HasAns_total': 28278, 'best_exact': 68.31459084800905, 'best_exact_thresh': 0.0, 'best_f1': 77.41966316717496, 'best_f1_thresh': 0.0}
```
```
 Results: {'exact': 80.23651844843897, 'f1': 87.79465245604408, 'total': 10570, 'HasAns_exact': 80.23651844843897, 'HasAns_f1': 87.79465245604408, 'HasAns_total': 10570, 'best_exact': 80.23651844843897, 'best_exact_thresh': 0.0, 'best_f1': 87.79465245604408, 'best_f1_thresh': 0.0}
```
```python=
squad__v2.4.json
Results: {'exact': 80.44465468306528, 'f1': 88.22124106174977, 'total': 10570, 'HasAns_exact': 80.44465468306528, 'HasAns_f1': 88.22124106174977, 'HasAns_total': 10570, 'best_exact': 80.44465468306528, 'best_exact_thresh': 0.0, 'best_f1': 88.22124106174977, 'best_f1_thresh': 0.0}
```
Problems
----
* Predict (dev_v1.1.json)
```python=1
Results: {'exact': 59.23368022705771,'f1': 68.35083996742, 'total': 10570, 'HasAns_exact': 59.23368022705771, 'HasAns_f1': 68.35083795996742, 'Has_total': 10570, 'best_exact': 59.23368022705771, 'best_exact_thresh': 0.0, 'best_f1': 68.35085996742, 'best_f1_thresh': 0.0}
```
* With pretrained model
```python=1
Results: {'exact': 0.34058656575212864, 'f1': 7.720321942106521, 'total': 10570, 'HasAns_exact': 0.34058656575212864, 'HasAns_f1': 7.720321942106521, 'HasAns_total': 10570, 'best_exact': 0.34058656575212864, 'best_exact_thresh': 0.0, 'best_f1': 7.720321942106521, 'best_f1_thresh': 0.0}
```
* Predict (my own dataset)
```python=1
Results: {'exact': 41.178584714717175, 'f1': 49.02356175328484, 'total': 158597, 'HasAns_exact': 56.73550595574245, 'HasAns_f1': 67.54524212535048, 'HasAns_total': 115099, 'NoAns_exact': 0.013793737643110027, 'NoAns_f1': 0.013793737643110027, 'NoAns_total': 43498, 'best_exact': 41.174801541012755, 'best_exact_thresh': 0.0, 'best_f1': 49.01977857957855, 'best_f1_thresh': 0.0}
```

Dataset
----
* 247
1. sqaud
> training data : train-v1.1.json + dev-v1.1.json(answer為詞眼過濾，question 為answer 放進 QG model) + dev-v1.1.json
2. squad1
> training data : train-v1.1.json + dev-v1.1.json(answer為詞眼過濾，question 為answer 放進 QG model) + dev-v1.1.json(皆為經過hugging face qa 過濾)
3. sqaud2
> training data : train-v1.1.json + dev-v1.1.json(answer為詞眼過濾，question 為answer 放進 QG model) + dev-v1.1.json(皆為經過hugging face qa 過濾)
4. squad3
> training data : train_v1.1.json + train_v3.0.json (留下與qa model 一樣的答案)

> result : Results: {'exact': 80.84200567644277, 'f1': 88.34102154018977, 'total': 10570, 'HasAns_exact': 80.84200567644277, 'HasAns_f1': 88.34102154018977, 'HasAns_total': 10570, 'best_exact': 80.84200567644277, 'best_exact_thresh': 0.0, 'best_f1': 88.34102154018977, 'best_f1_thresh': 0.0}
5. squad5
> training data : squad_v3.3.json(train-v3.3.json + train-v1.1.json)

> Results: {'exact': 80.1135288552507, 'f1': 87.88358745952684, 'total': 10570, 'HasAns_exact': 80.1135288552507, 'HasAns_f1': 87.88358745952684, 'HasAns_total': 10570, 'best_exact': 80.1135288552507, 'best_exact_thresh': 0.0, 'best_f1': 87.88358745952684, 'best_f1_thresh': 0.0}
6. squad6

(1)測試資料使用自己的考眼
> training data: train-v1.1.json

> evaluate file : dev-v1.1.json(自己的考眼)

> Results: {'exact': 54.421232426750755, 'f1': 67.29479069238222, 'total': 7611, 'HasAns_exact': 54.421232426750755, 'HasAns_f1': 67.29479069238222, 'HasAns_total': 7611, 'best_exact': 54.421232426750755, 'best_exact_thresh': 0.0, 'best_f1': 67.29479069238222, 'best_f1_thresh': 0.0}

(2)測試資料使用官方dev-v1.1.json
> Results: {'exact': 80.66225165562913, 'f1': 88.37611423613609, 'total': 10570, 'HasAns_exact': 80.66225165562913, 'HasAns_f1': 88.37611423613609, 'HasAns_total': 10570, 'best_exact': 80.66225165562913, 'best_exact_thresh': 0.0, 'best_f1': 88.37611423613609, 'best_f1_thresh': 0.0}
7. squad7
> training data : train-v1.1.json + train-v3.3_rand.json(隨機選擇六千筆)
> 
> Results: {'exact': 80.52034058656575, 'f1': 88.21159376037637, 'total': 10570, 'HasAns_exact': 80.52034058656575, 'HasAns_f1': 88.21159376037637, 'HasAns_total': 10570, 'best_exact': 80.52034058656575, 'best_exact_thresh': 0.0, 'best_f1': 88.21159376037637, 'best_f1_thresh': 0.0}
8. squad8
> training data: train-v1.1.json + train-v3.4.json(將q的品質較好的前六千筆作為資料)

> Results: {'exact': 80.59602649006622, 'f1': 88.2671292604995, 'total': 10570, 'HasAns_exact': 80.59602649006622, 'HasAns_f1': 88.2671292604995, 'HasAns_total': 10570, 'best_exact': 80.59602649006622, 'best_exact_thresh': 0.0, 'best_f1': 88.2671292604995, 'best_f1_thresh': 0.0}
9. squad9
> training data: train-v1.1.json + train-v3.3_len.json(將平均長度前六千筆作為訓練資料)

>  Results: {'exact': 80.52980132450331, 'f1': 88.19517709908695, 'total': 10570, 'HasAns_exact': 80.52980132450331, 'HasAns_f1': 88.19517709908695, 'HasAns_total': 10570, 'best_exact': 80.52980132450331, 'best_exact_thresh': 0.0, 'best_f1': 88.19517709908695, 'best_f1_thresh': 0.0}


* 252 
1. squad1
> training data : train_v1.1.json + train_v3.1.json(留下與qa model不同的答案)

> result : Results: {'exact': 79.85808893093662, 'f1': 87.62966929999214, 'total': 10570, 'HasAns_exact': 79.85808893093662, 'HasAns_f1': 87.62966929999214, 'HasAns_total': 10570, 'best_exact': 79.85808893093662, 'best_exact_thresh': 0.0, 'best_f1': 87.62966929999214, 'best_f1_thresh': 0.0}

2. squad2
> training data : train_v1.1.json + train_v3.2.json(考眼但無過濾)

> result : Results: {'exact': 80.34058656575213, 'f1': 88.08386369953621, 'total': 10570, 'HasAns_exact': 80.34058656575213, 'HasAns_f1': 88.08386369953621, 'HasAns_total': 10570, 'best_exact': 80.34058656575213, 'best_exact_thresh': 0.0, 'best_f1': 88.08386369953621, 'best_f1_thresh': 0.0}

3. squad3
> training data : train_3.3.json(只使用我的考眼作為訓練資料)