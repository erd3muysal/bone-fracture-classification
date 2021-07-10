# !/usr/bin/python
import config
import load
import model
import train
import test
import visualize
import argparse
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # Check the help parameters to understand arguments
    parser = argparse.ArgumentParser(description='Bone Fracture Classification')
    parser.add_argument('--trainDir', type=str, required=False, help="Train dataset directory path")
    parser.add_argument('--imageWidth', type=int, required=False, help="Image width")
    parser.add_argument('--imageHeight', type=int, required=False, help="Image height")
    parser.add_argument('--segmentation', type=bool, required=False, help="Segmentation selection")
    parser.add_argument('--configuration', type=str, required=False, help="Configuration selection")
    args = parser.parse_args()

    config.TRAIN_DIR = args.trainDir
    config.IMAGE_SIZE = (args.imageWidth, args.imageHeight)
    
    # Load train data
    X_path, y = load.load_paths(root_path=args.trainDir)  # Dizin adreslerini yükle
    X_path, y = shuffle(X_path, y)  # Verileri karıştır
    X_path_and_y = list(zip(X_path, y))  # Resmin dizin adresini ve ona ait etiketi birbirine bağla
    X_path = X_path[:100]  # Ram problemi yaşamamak için sadece 1000 tanesini kullan
    y = y[:100]
    X = load.load_images(X_path, (args.imageWidth, args.imageHeight), segmentation=args.segmentation, configuration=args.configuration)  # Resimleri yükle
    #visualize.plot_image_slices(X_path_and_y[:17]) 
    
    '''
    # Load test data
    X_test_path, y_test = load.load_paths(root_path=config.TEST_DIR)  # Dizin adreslerini yükle
    X_test_path, y_test = shuffle(X_test_path, y_test)  # Verileri karıştır
    X_test_path_and_y_test = list(zip(X_test_path, y_test))  # Resmin dizin adresini ve ona ait etiketi birbirine bağla
    X_test_path = X_test_path[:250]  # Ram problemi yaşamamak için sadece 1000 tanesini kullan
    y_test = y_test[:250]
    X_test = load.load_images(X_test_path, config.IMAGE_SIZE, segmentation=False, configuration='pretrained')  # Resimleri yükle
    '''
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=config.SEED)  # Eğitim ve test verilerini oluştur
    y_train = to_categorical(y_train, num_classes=2)
    y_valid = to_categorical(y_valid, num_classes=2)
    #y_test = to_categorical(y_test, num_classes=2)

    print("---------------------------------------")
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", y_train.shape)
    print("---------------------------------------")
    print(X_train.shape[0], "train samples")
    print(X_valid.shape[0], "test samples")

    model = model.build_ann_model()
    history = train.train(model=model, X_=X_train, y_=y_train, num_epochs=5, batch_size=8)

#    y_pred = model.predict(X_test, verbose=1)
#    pred_decoded = [train.decode_label(pred) for pred in y_pred]
#    loss, score = model.evaluate(x=X_test, y=y_test)
#    print("VGG16 Test Başarımı:", score)