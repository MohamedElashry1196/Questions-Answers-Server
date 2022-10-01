import os
from flask import Flask, render_template, request , jsonify 
from transformers import pipeline
import pandas as pd
import json
import time

qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/bert-small-finetuned-squadv2",
    tokenizer="mrm8488/bert-small-finetuned-squadv2"
)

def Data_cleaning (data):
  data = eval(repr(data))
  data = json.loads(data, strict=False)
  df = pd.DataFrame.from_dict(data)
  cols = ["text","question","answer"]
  comp_list = []
  for index, row in df.iterrows():
    for i in range(len(row["data"]["questions"])):
        temp_list = []
        temp_list.append(row["data"]["story"])
        temp_list.append(row["data"]["questions"][i]["input_text"])
        temp_list.append(row["data"]["answers"][i]["input_text"])
        comp_list.append(temp_list)
  
  new_df = pd.DataFrame(comp_list, columns=cols) 
  
  return new_df 

def answer_question(question, text, qa_pipeline=qa_pipeline):
    '''
    Takes a `question` string and an `text` string and tries to identify 
    the words within the `answer` that can answer the question. Prints them out.
    '''
    
    result = qa_pipeline({'context': text,
    'question': question})
    
    return result["answer"],result["score"]

def test_score(df):
    matched = 0 
    for ind in df.index:
        print(ind)
        result = answer_question(df['question'][ind], df['text'][ind])
        act_answer = df['answer'][ind]
        prd_answer = result[1]
        print(act_answer)
        print(prd_answer)
        if str(act_answer) in str(prd_answer) or str(prd_answer) in str(act_answer) :
            matched+=1
            print("matched : " , matched )
    
    score = matched/len(df.index)
    
    return score

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def index():
    result = []
    form = request.get_json()
    bert_abstract = form['paragraph']
    question = form['question']
    result.append(form['question'])
    answer,score = answer_question(question, bert_abstract)
    result.append(form['paragraph'])
    result.append(answer)
    result.append(score)
    
    return jsonify({'result' : result})

@app.route('/score', methods=['POST'])
def score():
    result = []
    # form = request.get_json()
    # sample_dataset = str(form['sample']).encode('utf-8')
    # df = Data_cleaning(sample_dataset)
    # score = test_score(df)
    # result.append(form['sample'])
    # result.append(score)
    form =  request.form
    sample_dataset = str(form['sample']).encode('utf-8')
    df = Data_cleaning(sample_dataset)
    print(df)
    score = test_score(df)
    print(score)
    result.append(form['sample'])
    result.append(score)
    
    return jsonify({'result' : result})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
