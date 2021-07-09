import os

MODEL_PATH = '../results/models/model.13-0.79.h5'
IMAGE_SIZE = (224, 224)
MEAN = 51.43062
STD = 44.75102
LABEL_DICT = {0: 'Bone Fracture Negative', 1: 'Bone Fracture Positive'}
IMAGE_UPLOADS = os.path.join(os.getcwd(), 'static/uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
