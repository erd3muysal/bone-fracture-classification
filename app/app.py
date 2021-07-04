#!/usr/bin/env python
import os
import numpy as np
from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from PIL import Image


MODEL_PATH = '../results/models/model.13-0.79.h5'
IMAGE_SIZE = (224, 224)
MEAN = 51.43062
STD = 44.75102
LABEL_DICT = {0: 'Bone Fracture Negative', 1: 'Bone Fracture Positive'}
UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = load_model(MODEL_PATH)


@app.route('/', methods=['GET'])
def index():
	return render_template('index.html')

def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		# If the user does not select a file, the browser submits an
		# empty file without a filename.
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			img = preprocess(file_path)
			result = predict(img)
			return result
	return '''
	<!doctype html>
	<title>Upload new File</title>
	<h1>Upload new File</h1>
	<form method=post enctype=multipart/form-data>
	  <input type=file name=file>
	  <input type=submit value=Upload>
	</form>
	'''

def predict(img):
	# Make prediction
	y_pred = model.predict(img)
	label = decode_label(y_pred[0][0])
	result = LABEL_DICT[label]
	return result

def preprocess(file_path):
	img = load_img(file_path, color_mode='grayscale', target_size=IMAGE_SIZE)
	img = img_to_array(img)
	img = np.asarray(img).astype('float32')
	img = np.squeeze(img, axis=-1)
	img = np.repeat(img[..., np.newaxis], 3, -1)
	img = img[np.newaxis, ...]
	img = (img - MEAN) / STD

	return img

def decode_label(y_pred):
	return 1 if y_pred > 0.5 else 0