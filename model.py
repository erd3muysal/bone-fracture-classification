# !/usr/bin/python
import config
import tensorflow as tf
import keras
import keras.layers as layers
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPool2D, BatchNormalization, GlobalAveragePooling2D, Dropout
from keras.models import Sequential, Model
from keras.regularizers import l1, l2, l1_l2
#from keras.applications import InceptionV3, VGG16, DenseNet121, DenseNet169, DenseNet201, ResNet50, ResNet101, MobileNet, MobileNetV2
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16


def ResNet50_model():
	'''
	Build an InceptionV3 mdeol with ImageNet weights.

	returns:
	model -- Model
	'''

	# Load InceptionV3 model and remove last layer.
	base_model = ResNet50(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
	base_model.trainable = False
	tail = base_model.output
	x = GlobalAveragePooling2D()(tail)
	#x = Flatten()(tail)
	x = Dense(units=1024, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
	x = Dense(units=512, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
	x = Dense(units=256, activation='relu')(x)	
	x = Dropout(0.3)(x)

	outputs = layers.Dense(units=2, activation='softmax')(x)
  
	# Define the model
	model = Model(inputs=base_model.input, outputs=outputs, name='ResNet50')

	return model


def build_ann_model():
	'''
	4 katmanlı yapay sinir ağı modeli inşa eder.
	'''

	inputs = keras.Input(shape=config.INPUT_SHAPE)  # Giriş katmanı

	x = layers.Flatten()(inputs)  # (512, 512, 1) boyutunda olan resimleri (512x512, 1) = (262144, 1) haline getirir. Yapay sinir ağları için bu formatta olmalı girdiler.
	x = layers.Dense(units=64, activation='relu')(x)  # 1. gizli katman: 64 nöron + relu
	x = layers.Dense(units=32, activation='relu')(x)  # 2. gizli katman: 32 nöron + relu
	x = layers.Dense(units=16, activation='relu')(x)  # 3. gizli katman: 16 nöron + relu

	outputs = layers.Dense(units=2, activation='sigmoid')(x)  # Çıkış katmanı: 1 nöron + sigmoid

	# Define the model.
	model = keras.Model(inputs=inputs, outputs=outputs, name='yapay_sinir_aglari')

	return model
