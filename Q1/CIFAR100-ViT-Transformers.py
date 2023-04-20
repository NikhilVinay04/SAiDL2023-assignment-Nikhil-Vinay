import sys
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Softmax
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from copy import copy
import numpy as np
import tensorflow as tf
from transformers import TFViTModel
from vit_keras import vit, utils
import tensorflow_addons as tfa

input_shape = (32, 32, 3) #Cifar100 image size
image_size = 256 #size after resizing image
num_classes = 100

def modelMetrics(testX,testY,testActualClasses,yhat_classes) :    
    accuracy=accuracy_score(testActualClasses,yhat_classes)
    print('Accuracy score: %f' % accuracy)

    precision=precision_score(testActualClasses,yhat_classes,average='weighted')
    print('Precision score: %f' % precision)

    recall=recall_score(testActualClasses,yhat_classes,average='weighted')
    print('Recall score: %f' % recall)

    f1=f1_score(testActualClasses,yhat_classes,average='weighted')
    print('F1 score : %f' % f1)
     
    kappa = cohen_kappa_score(testActualClasses, yhat_classes)
    print('Cohens Kappa score: %f' % kappa)

    matrix = confusion_matrix(testActualClasses, yhat_classes)
    print('Confusion Matrix :')
    print(matrix)    

    return

def plotgraphs(his) :
    pyplot.suptitle('Loss and Accuracy Plots', fontsize=18)

    pyplot.subplot(1,2,1)
    pyplot.plot(his.history['loss'], label='Training Loss')
    pyplot.plot(his.history['val_loss'], label='Validation Loss')
    pyplot.legend()
    pyplot.xlabel('Number of epochs', fontsize=15)
    pyplot.ylabel('Loss', fontsize=15)

    pyplot.subplot(1,2,2)
    pyplot.plot(his.history["accuracy"], label='Train Accuracy')
    pyplot.plot(his.history["val_accuracy"], label='Validation Accuracy')
    pyplot.legend()
    pyplot.xlabel('Number of epochs', fontsize=14)
    pyplot.ylabel('Accuracy', fontsize=14)
    pyplot.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=0.4,hspace=0.4)
    pyplot.show()

    return


def load_dataset() :
    (trainX,trainY),(testX,testY) = cifar100.load_data()
    trainX,trainValX,trainY,trainValY=train_test_split(trainX,trainY,test_size=0.3)
    
    print ('Train : X=%s, Y=%s'%(trainX.shape,trainY.shape))
    print ('Validation : X=%s, Y=%s'%(trainValX.shape,trainValY.shape))    
    print ('Test : X=%s, Y=%s'%(testX.shape,testY.shape))
#    for i in range(9) :
#        pyplot.subplot(30,1,i+1)
#        pyplot.imshow(trainX[i])
#    pyplot.show()

    testActualClasses=testY.flatten()

    trainY = to_categorical(trainY)
    trainValY = to_categorical(trainValY)    
    testY = to_categorical(testY)
    
    return trainX,trainY,trainValX,trainValY,testX,testY,testActualClasses
    
def prep_pixels(train,trainVal,test) :
    train_norm = train.astype('float32')
    trainVal_norm = trainVal.astype('float32')
    test_norm = test.astype('float32')
    
    train_norm = train_norm / 255.0
    trainVal_norm = trainVal_norm / 255.0
    test_norm = test_norm / 255.0
    
    return train_norm,trainVal_norm,test_norm


def Vision_Transformer_model():
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (image_size, image_size)))(inputs) #Resize image to  size 224x224
    base_model = vit.vit_b16(image_size=image_size, activation="sigmoid", pretrained=True, include_top=False, pretrained_top=False)
    
    base_model.trainable = False #Set false for transfer learning
    x = base_model(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation=tfa.activations.gelu)(x)
    x = BatchNormalization()(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def run_test_harness_using_transformers(model):
    trainX,trainY,trainValX,trainValY,testX,testY,testActualClasses = load_dataset()
    trainX,trainValX,testX = prep_pixels(trainX,trainValX,testX)

    batch_size = 32
    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_generator = datagen.flow(trainX, trainY, batch_size=batch_size)

    opt = SGD(learning_rate=0.01,momentum=0.9)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

    his = model.fit(trainX,trainY,batch_size=batch_size,epochs=2,validation_data=(trainValX,trainValY))

    for layer in model.layers:
      layer.trainable = True

    opt = SGD(learning_rate=0.001,momentum=0.9)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

    his = model.fit(trainX,trainY,batch_size=batch_size,epochs=5,validation_data=(trainValX,trainValY))

    plotgraphs(his)

    _,acc = model.evaluate(testX,testY,verbose=1)
    print('Accuracy for test data is >%.3f'%(acc*100.0))

    yhat_probs = model.predict(testX, verbose=0)
    yhat_classes = np.argmax(yhat_probs,axis=-1)

    modelMetrics(testX,testY,testActualClasses,yhat_classes)

    return


vision_transformer = Vision_Transformer_model()
run_test_harness_using_transformers(vision_transformer)
