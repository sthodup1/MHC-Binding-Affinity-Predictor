import scipy.io
import numpy as np
import random
import sys, os
import h5py
import keras.backend as K
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda, Reshape
from keras.layers.merge import maximum
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras import layers
from keras import regularizers
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import seaborn as sns
sns.set(style='ticks', palette='Set2')
sns.set_context("poster")


np.random.seed(1) # for reproducibility

hla_type = sys.argv[1]
Padded = sys.argv[2]
n_classes = int(sys.argv[3])

plot_name = './CNN_plots/' 
if Padded == 'True':
    X_train = np.load('Data/'+hla_type+'_padded/'+'Xtrain.npy')
    Y_train = np.load('Data/'+hla_type+'_padded/'+'Ytrain.npy')
    X_test = np.load('Data/'+hla_type+'_padded/'+'Xtest.npy')
    Y_test = np.load('Data/'+hla_type+'_padded/'+'Ytest.npy')
    X_valid = np.load('Data/'+hla_type+'_padded/'+'Xval.npy')
    Y_valid = np.load('Data/'+hla_type+'_padded/'+'Yval.npy')
    core = 'Data/'+ hla_type+'_padded'
    fname = 'CNN_padded_segment_results.txt'
    plot_name += 'padded/' + hla_type + '_'
else:
    X_train = np.load('Data/'+hla_type+'_truncated/'+'Xtrain.npy')
    Y_train = np.load('Data/'+hla_type+'_truncated/'+'Ytrain.npy')
    X_test = np.load('Data/'+hla_type+'_truncated/'+'Xtest.npy')
    Y_test = np.load('Data/'+hla_type+'_truncated/'+'Ytest.npy')
    X_valid = np.load('Data/'+hla_type+'_truncated/'+'Xval.npy')
    Y_valid = np.load('Data/'+hla_type+'_truncated/'+'Yval.npy')
    core = 'Data/'+ hla_type+'_truncated'
    fname = 'CNN_truncated_segment_results.txt'
    plot_name += 'truncated/' + hla_type + '_'

plot_name += str(n_classes) + '.png'

X_train = np.transpose(X_train,axes=(2, 1, 0))
X_valid = np.transpose(X_valid,axes=(2, 1, 0))
X_test = np.transpose(X_test,axes=(2, 1, 0))




y_test = np.zeros((np.shape(Y_test)[0]))
for i in range(np.shape(Y_test)[0]):
    for j in range(n_classes):
        if(Y_test[i][j] == 1):
            y_test[i] = j


print ('X_train:',X_train.shape)
print ('Y_train:',Y_train.shape)
print ('X_test:',X_test.shape)
print ('Y_test:',Y_test.shape)
print ('X_valid:',X_valid.shape)
print ('Y_valid:',Y_valid.shape)

train_shape = X_train.shape

peptide_length = train_shape[1]
print(peptide_length)



model = Sequential()

kern_sC1 = 2
filterC1 = 200

model.add(Conv1D(activation="relu",
                        input_shape=(peptide_length, 24),
                        filters=filterC1,
                        kernel_size=kern_sC1,
                        padding="valid",
                        strides=1))


model.add(Dropout(0.3))

model.add(MaxPooling1D(pool_size=peptide_length-kern_sC1+1, strides=None))

model.add(Flatten())

model.add(Dense(units=filterC1))
model.add(Activation('relu'))

model.add(Dense(units=10))
model.add(Activation('relu'))

model.add(Dense(units=n_classes))
model.add(Activation('sigmoid'))

print(model.summary())
print('compiling model')
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

print('running at most 500 epochs')

checkpointer = ModelCheckpoint(filepath=core+"/bestmodelCNN.hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=30, verbose=1)

model.fit(X_train, Y_train, batch_size=100, epochs=200, callbacks=[checkpointer,earlystopper], validation_data=[X_valid, Y_valid], shuffle=True)

tresults = model.evaluate(X_test, Y_test)

print(tresults)

y_pred = model.predict_classes(X_test, verbose=0)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """


    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    ax = sns.heatmap(cm,vmin=0, vmax=1)
    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, cm[i, j],
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix

# Plot normalized confusion matrix
plt.figure()
title = 'Normalized Confusion Matrix With ' + str(n_classes) + ' Classes'
plot_confusion_matrix(cnf_matrix, classes=range(n_classes), normalize=True,
                      title=title)

# plt.show()
plt.savefig(plot_name)
