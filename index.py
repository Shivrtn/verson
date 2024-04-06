#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras
import numpy as np
import cv2
model=keras.models.load_model('footbal_cat_model.keras')
from flask import Flask, request, jsonify
import os
from flask_cors import CORS
fold={"Karshan":0,"Na":1,"Mi":2,"Ma":3,"cycle":4,"football":5,"cat":6}
app = Flask(__name__)
CORS(app) 
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'fileToUpload' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['fileToUpload']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filestr = file.read()  
        file_bytes = np.fromstring(filestr, np.uint8)
        img = np.array(cv2.resize(cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED),(50,50))).reshape(1,50,50,3)
        prediction=("predicted",list(fold.keys())[(model.predict(img).argmax())])
        print(prediction)
        return jsonify({'message': prediction}), 200
if __name__ == '__main__':
    app.run( port=3000)

    

