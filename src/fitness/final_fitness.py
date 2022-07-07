from inspect import Parameter
from random import shuffle
from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
import numpy as np
import re
import pandas as pd
import numpy as np 
import itertools
from sklearn.metrics import f1_score
import tensorflow.keras
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from tensorflow.keras.models import Sequential 
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dropout, Flatten, Dense, AveragePooling2D, Conv2D, MaxPool2D 
from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import math  
import datetime
import time
from tensorflow.keras import optimizers, callbacks
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16

#####imports#######

import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop






class final_fitness(base_ff):
    """Fitness function for matching a string. Takes a string and returns
    fitness. Penalises output that is not the same length as the target.
    Penalty given to individual string components which do not match ASCII
    value of target."""
    maximise = True

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        maximise = True

    def evaluate(self, ind, **kwargs):
        with tf.device('/gpu:0'):
            guess = str(ind.phenotype)
            clean = re.sub('[^A-Za-z0-9]+\.[0-9]+', ' ', guess)
            a = clean.split()

            

            train_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
            test_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
                        

            
            train_generator = train_datagen.flow_from_directory('output1/train',target_size=(224, 224),class_mode='categorical', batch_size=int(a[2]))  # since we use binary_crossentropy loss, we need binary labels
            test_generator = train_datagen.flow_from_directory('output1/test',target_size=(224, 224),class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels
            

            input_tensor = Input(shape=(224, 224, 3))

            model1 = keras.applications.VGG16(weights='imagenet', include_top=True, input_tensor=input_tensor) 

            for layers in (model1.layers)[: int (len(model1.layers)*0.7)]:
                layers.trainable = False # change code

            predictions = Dense(7, activation="softmax")(x)
            model_final = Model(inputs=[model1.input],outputs=[predictions])

            if str(a[0]) == 'adam':
                opt = Adam(lr=float(a[1]))
            elif str(a[0]) == 'rmsprop':
                 opt = RMSprop(lr=float(a[1]))

        

            model_final.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics=["accuracy"])
            hist = model_final.fit_generator(generator= train_generator, 
                                 epochs= 10, validation_data= test_generator, steps_per_epoch = 26250 // int(a[2]),
                            validation_steps = 8750 // int(a[2] )   
                                 )
            f1 = hist.history['val_accuracy']

        return f1   


