#!/usr/bin/env python
import os
import numpy as np
from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import config

def create_app(test_config=None):
	# create and configure the app
	app = Flask(__name__, instance_relative_config=True)
	# load the instance config, if it exists, when not testing

	# ensure the instance folder exists
	try:
		os.makedirs(config.IMAGE_UPLOADS)
	except OSError:
		pass


	@app.route('/', methods=['GET'])
	def index():
		return render_template('index.html')

	def allowed_file(filename):
		return '.' in filename and \
				filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

	@app.route('/predict', methods=['GET', 'POST'])
	def upload_file():
		if request.method == 'POST':
			# Check if the post request has the file part
			if 'file' not in request.files:
				flash('No file part')
				return redirect(request.url)
			
			file = request.files['file']
			# If the user does not select a file, the browser submits and
			# empty file without a filename.
			if file.filename == '':
				flash('No selected file')
				return redirect(request.url)

			if file and allowed_file(file.filename):
				filename = secure_filename(file.filename)
				file.save(os.path.join(config.IMAGE_UPLOADS, filename))
				print('XXXXX', os.path.join(config.IMAGE_UPLOADS, filename))
				img = preprocess(os.path.join(config.IMAGE_UPLOADS, filename))
				pred = predict(img)
				return pred
		return '''
			<h1>Upload new File</h1>
			<form method="post" enctype="multipart/form-data">
			<input type="file" name="file1">
			<input type="submit">
			</form>
			'''

	def predict(img):
		# Make prediction
		y_pred = load_model(config.MODEL_PATH).predict(img)
		label = decode_label(y_pred[0][0])
		result = config.LABEL_DICT[label]
		return result

	def preprocess(file_path):
		img = load_img(file_path, color_mode='grayscale', target_size=config.IMAGE_SIZE)
		img = img_to_array(img)
		img = np.asarray(img).astype('float32')
		img = np.squeeze(img, axis=-1)
		img = np.repeat(img[..., np.newaxis], 3, -1)
		img = img[np.newaxis, ...]
		img = (img - config.MEAN) / config.STD
		return img

	def decode_label(y_pred):
		return 1 if y_pred > 0.5 else 0

	return app