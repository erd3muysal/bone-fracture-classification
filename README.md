# Bone Fracture Classification with Machine Learning
This notebook includes implementation of several machine learning techniques in order to solve bone fracture classification problem by determining whether an X-ray study is normal or abnormal.
## Data
Dataset that used in this project is called [MURA](https://stanfordmlgroup.github.io/competitions/mura/) which is a large dataset of bone X-rays.
## Models
Classification Methods:
- KNN (sklearn).
- ANNs (tf.keras), vanilla and pretrained models.
## Dependencies
```
absl-py==0.12.0
astunparse==1.6.3
cachetools==4.2.2
certifi==2020.12.5
chardet==4.0.0
click==8.0.0
Flask==2.0.0
flatbuffers==1.12
gast==0.3.3
google-auth==1.30.0
google-auth-oauthlib==0.4.4
google-pasta==0.2.0
grpcio==1.32.0
h5py==2.10.0
idna==2.10
itsdangerous==2.0.0
Jinja2==3.0.0
Keras==2.4.3
Keras-Preprocessing==1.1.2
Markdown==3.3.4
MarkupSafe==2.0.0
numpy==1.19.5
oauthlib==3.1.0
opt-einsum==3.3.0
Pillow==8.2.0
protobuf==3.17.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
PyYAML==5.4.1
requests==2.25.1
requests-oauthlib==1.3.0
rsa==4.7.2
scipy==1.6.3
six==1.15.0
tensorboard==2.5.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.0
tensorflow==2.4.1
tensorflow-estimator==2.4.0
termcolor==1.1.0
typing-extensions==3.7.4.3
urllib3==1.26.4
Werkzeug==2.0.0
wrapt==1.12.1
ps: All the libraries can be downloaded by pip install -r requirements.txt
```
## Installation Options
1.  Clone repo to your local:  `$ git clone https://github.com/erd3muysal/bone_fracture_classification`.
2. Go to project directory and create a virtual envrironment.
    * `$ python -m venv env`
    * `$ source env/bin/activate`
4. Install dependinces by typing `pip install -r requirements.txt`.
5.  Train the model. You have two options for training process.
    * Notebook
    or
    * Scripts
    This process will save the best model as a `.h5` file under `results/models` directory.
6. [under construction]
## Usage
`$ python3 app/app.py`
## Author
R. Erdem Uysal
* https://www.linkedin.com/in/erdemuysal13/
* https://github.com/erd3muysal
