# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 23:10:45 2018

@author: Sheetal
"""

from keras.models import Sequential
from keras.layers import Dense 
import pandas as pd
import numpy
import math
from keras.models import model_from_json


import keras
print(keras.__version__)

# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = pd.read_csv("diabetes.csv", delimiter=",") 

print(dataset.head(20))

# split into input (X) and output (Y) variables
array = dataset.values
X = array[:,0:8] 
Y = array[:,8] 

# create model 
model = Sequential() 
model.add(Dense(12, input_dim=8, activation='relu')) 
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 


# Fit the model 
model.fit(X, Y, epochs=150, batch_size=10)


# evaluate the model 
scores = model.evaluate(X, Y) 
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

## #serializing our model to a file called model.pkl - Not working for KERAS models
#import pickle

#pickle.dump(model, open("model.pkl","wb"))


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# later...
 
# load json and create model


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


## Predicting one record


df = pd.DataFrame([[1,85, 66, 29, 0, 26.6, 0.351, 31]], columns=list('ABCDEFGH'))
array=df.values


prediction=loaded_model.predict(array)
print(prediction)


## For testing purpose only (since Pycharm was throwing error on Prediction)

print(prediction.tolist())

response = {}
response['predictions'] = loaded_model.predict([array]).tolist()