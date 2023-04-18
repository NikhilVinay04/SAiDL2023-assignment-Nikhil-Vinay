import sys
from tensorflow import keras
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import numpy as np


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
    print('Cohens Kappa score %f' % kappa)

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
    
def cnn_stdsoftmax_model():
    model = Sequential()
    model.add(Conv2D(128,(3,3),activation='relu',kernel_initializer='he_uniform',padding='same',input_shape=(32,32,3)))
    model.add(Conv2D(128,(3,3),activation='relu',kernel_initializer='he_uniform',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024,activation='relu',kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(100,activation='softmax'))
    
    opt = SGD(learning_rate=0.01,momentum=0.9)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    return model
        
def run_test_harness_std_cnn() :
    trainX,trainY,trainValX,trainValY,testX,testY,testActualClasses = load_dataset()
    trainX,trainValX,testX = prep_pixels(trainX,trainValX,testX)

    model=cnn_stdsoftmax_model()

    his = model.fit(trainX,trainY,epochs=50,batch_size=64,validation_data=(trainValX,trainValY),verbose=1)
    plotgraphs(his)

    _,acc = model.evaluate(testX,testY,verbose=1)
    print('Accuracy for test data is >%.3f'%(acc*100.0))

    yhat_probs = model.predict(testX, verbose=0)
    yhat_classes = np.argmax(yhat_probs,axis=-1)

    modelMetrics(testX,testY,testActualClasses,yhat_classes)

    return
    

run_test_harness_std_cnn()   


