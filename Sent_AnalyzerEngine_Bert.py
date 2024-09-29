
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' #It prevents the program crashing due to detection of multiple copies of library.

import numpy as np
from flask import Flask,request,jsonify,render_template
import joblib
import pandas as pd
import datetime
import os

from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__) # it initialize the flask api

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Loading our trained bert model
custom_model = BertForSequenceClassification.from_pretrained('saved_models/bert_train_4ep')

def predictfunc(review):
    #prediction = classifier.predict(review)
    inputs = tokenizer(review, padding=True, truncation=True, return_tensors='pt').to('cpu')
    outputs = custom_model(**inputs) # **inputs is used to unpack all tensors within inputs and load it perfectly into custom model.

    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)  # It applies softmax function on the output.logits comes from custom model
    predictions = predictions.cpu().detach().numpy() # It converts resultant tensors to numpy array
    pred_label = np.argmax(predictions)
    if pred_label== 1:
        sentiment='Positive'
    else:
        sentiment='Negative'
    return pred_label,sentiment

@app.route('/')   # it creates route for the root url
def index():
    return render_template('New_home.html')  # it is the landing page of web application.

# flask has many methods like get,put,delete,options,etc. we can use any method as per our requirement.
#here we used 'post' method to allow user to post 'data' that could be processed by server.
@app.route('/predict', methods=['POST'])  # it defines route for 'predict' url that only takes 'post' request.
def predict():
     
     if request.method == 'POST':
        result = request.form
        content = request.form['review']
        review = str(content)
        prediction,sentiment =predictfunc(review)      
     return render_template("New_predict.html",pred= prediction, sent=sentiment) # it gives the prediction page

if __name__ == '__main__':
     #app.run(debug = True,port=8080)
     app.run(host='0.0.0.0')