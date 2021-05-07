import json
with open('eval_beam_size_1.json','r') as readfile:
    qa = json.load(readfile)
result = {} 
result['data'] = []
result['version'] = "1.1"
i = 0
total = 0
for q in qa:
    try :
        total += 1
        temp = {}
        temp_2 = {}
        text = q['context']
        answer = q['answers']
        question = q['gen_questions'][0]    
        temp['title'] = answer[0]['text']
        temp['paragraphs'] = [{}]
        temp['paragraphs'][0]['context'] = text
        temp['paragraphs'][0]['qas'] = []
        temp_2['answers'] = answer
        temp_2['question'] = question
        temp_2['id'] = str(total)
        temp['paragraphs'][0]['qas'].append(temp_2)
        if temp_2['answers'][0]['answer_start'] != -1:
            result['data'].append(temp)
            i += 1
    except KeyError:
        pass
print(total)
print("num = ",i)
with open('train_v3.0_raw.json','w') as writefile:
    json.dump(result,writefile)

