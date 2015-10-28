import nnet_ext

import csv as csv 
import numpy as np
import os
import copy
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

#==============================================================================
# Read training set (csv)
#==============================================================================
current_directory = os.getcwd()
csv_training = csv.reader(open(current_directory + '/training.csv', 'rb')) 
training = []
for row in csv_training:
    training.append(row)        
training = np.array(training, dtype=float)

#==============================================================================
# Read test set (csv)
#==============================================================================
csv_test = csv.reader(open(current_directory + '/test.csv', 'rb')) 
test = []
for row in csv_test:
    test.append(row)        
test = np.array(test, dtype=float)

print '+Read csv done!'

#==============================================================================
# Normalization and split
#==============================================================================
scaler          = preprocessing.StandardScaler().fit(training[:,[0,1]])
training_n      = scaler.transform(training[:,[0,1]])
X_tr_n, X_cv_n, y_tr, y_cv = train_test_split(training_n, training[:,[2]], test_size=0.5, random_state=0)
X_te, y_te = scaler.transform(test[:,[0,1]]), test[:,[2]]

#==============================================================================
# Classification
#==============================================================================
layers = [2,100,1]
NN = nnet_ext.nnet(layers)
NN.set_learning_rate(0.9)
NN.set_epochs(100)
NN.set_batchsize(1)
NN.set_loss("quadratic")
NN.set_eval("accuracy")
NN.set_lambda(0.0)
NN.set_random_state(0)
NN.fit(X_tr_n, y_tr, X_cv_n, y_cv)


prediction_tr = np.array(NN.predict(X_tr_n))
print "Train Accuracy: %0.4f" %(accuracy_score(y_tr, prediction_tr>0.5))
prediction_val = np.array(NN.predict(X_cv_n))
print "Val Accuracy: %0.4f" %(accuracy_score(y_cv, prediction_val>0.5))
prediction_te = np.array(NN.predict(X_te))
print "Test Accuracy: %0.4f" %(accuracy_score(y_te, prediction_te>0.5))

plt.figure(1)
plt.plot(prediction_te)
