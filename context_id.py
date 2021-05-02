#為squad context 增加id
import json
from tqdm import tqdm
import re
def remove_articles(t):
    #t = tokenize(t)
    return re.sub("\"","",t)

#open two file
with open('squad_v3.0.json','r') as file1:
    data1 = json.load(file1)
result1 = data1.copy()
with open('train-v1.1.json','r') as file2:
    data2 = json.load(file2)
result2 = data2.copy()
id_num = 1
i = 0
flag = False
for d in tqdm(result1['data']):
    context = d['paragraphs'][0]['context']
    context_list = context.split(' ',3)[0:2]
    #print(context_list[0:2])
    for data in result1['data']:
        for p in data['paragraphs']:
            text = p['context']
            word = text.split(' ',3)[0:2]
            if word == context_list:
                p['context_id'] = id_num
    for data in result2['data']:
        for p in data['paragraphs']:
            text = p['context']
            word = text.split(' ',3)[0:2]
            if word == context_list:
                p['context_id'] = id_num
    id_num += 1
    '''for dd in result2['data']:
        for p in dd['paragraphs']:
            i += 1
            p_list = p['context'].split(' ',3)
            #print(context_list[0] , p_list[0])
            if context_list[0:2] == p_list[0:2] :
                p['context_id'] = id_num
                #print("id =",id_num)
                #print(context_list[0:2])
                d['paragraphs'][0]['context_id'] = id_num
                id_num += 1
                #print(p)
                break'''
            
print(i)
with open('squad_v3.0_pred.json','w') as write_file1:
    json.dump(result1,write_file1)
with open('squad_v3.0_data.json','w') as write_file2:
    json.dump(result2,write_file2)
