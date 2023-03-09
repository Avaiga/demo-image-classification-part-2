import tensorflow as tf
from tensorflow.keras import layers, models  
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
from tensorflow.keras.utils import to_categorical  
import pandas as pd   
import numpy as np

class_names =  ['AIRPLANE', 'AUTOMOBILE', 'BIRD', 'CAT', 'DEER', 'DOG', 'FROG', 'HORSE', 'SHIP', 'TRUCK']

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train / 255.0
y_train = to_categorical(y_train, len(class_names))
x_test = x_test / 255.0
y_test = to_categorical(y_test, len(class_names))

def tf_read(path: str): return tf.keras.models.load_model(path)
def tf_write(model, path: str):model.save(path)

#Task 1.1: Building the base model
def initialize_model(loss_f):
  # Creating model base
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',))
    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',))
    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                    loss=loss_f,
                    metrics=['accuracy'])

    return model

#Task 1.2: Initial training witha fixed number of epochs
datagen = ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=3/32,
    height_shift_range=3/32
)

def initial_model_training(n_epochs, model):
    print("INITIAL MODEL TRAINING STARTED: ")

    h = model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=n_epochs,
        validation_data=(x_test, y_test))

    training_result = pd.DataFrame.from_dict(h.history)
    training_result["N_Epochs"] = range(1,len(training_result)+1)
    
    return training_result, model

#Task 2.1: Merge train with a chosen number of epochs (training + validation set as training)
def merged_train(number_of_epochs,model):
    print("MERGED TRAIN STARTED: ")
    # merge the training and validation sets
    x_all = np.concatenate((x_train, x_test))
    y_all = np.concatenate((y_train, y_test))

    h = model.fit(
        datagen.flow(x_all, y_all, batch_size=64),
        epochs=number_of_epochs)
    
    training_result = pd.DataFrame.from_dict(h.history)
    training_result["N_Epochs"] = range(1,len(training_result)+1)
    
    return training_result, model

#Task 2.2: Predict image class
def predict_image(image_path, trained_model):
    print("PREDICTION TASK STARTED: ")
    img_array = tf.keras.utils.load_img(image_path, target_size=(32, 32))
    image = tf.keras.utils.img_to_array(img_array)  
    image = np.expand_dims(image, axis=0) / 255. 
    prediction_result = class_names[np.argmax(trained_model.predict(image))]
    print("Prediction result: {}".format(prediction_result))
    return prediction_result

