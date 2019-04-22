#  based on this repo's mnist setup for keras
#  https: // github.com/keras-team/keras/blob/master/examples/mnist_cnn.py


from __future__ import print_function
import random
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

## GLOBAL CONSTANTS

## FROM MNIST
# batch_size = 128
num_classes = 10
# epochs = 12

## FROM REUTERS
# Do we wanna keep these params?
# max_words = 1000
batch_size = 32
epochs = 5
pop_size = 5  # change this to 3
lambda_ = 1
sigma = 1.0
num_gens = 20
print_rate = 5

# GLOBAL PARAMETERS
act_list = ['tanh', 'softmax', 'elu', 'selu', 'softplus', 'softsign',
            'relu', 'sigmoid', 'hard_sigmoid', 'exponential', 'linear']
default_act = 'linear'
default_kernel = (1,1)
kernel_vals = range(1,28/2)
parent_pop = []
parent_pop_evaluated = []
elites = []

# DATA PROCESSING SETUP

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)





# original:
#   model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Dropout(0.25))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# [kernelsize, 1st activation, 1st dropout, 2nd activation, 2nd dropout, 3rd activation]
def makeModel(network):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=network[0],
                    activation=network[1],
                    input_shape=input_shape))
    model.add(Dropout(network[2]))
    model.add(Dense(128, activation=network[3]))
    model.add(Dropout(network[4]))
    model.add(Dense(num_classes, activation=network[5]))
    return model


def evaluateNetworks(parentPop):
    evaled = []
    for network in parentPop:
        currModel = makeModel(network)
        modelScore = trainModel(currModel,  x_train, y_train, x_test, y_test)
        evaled.append((network, modelScore))
    return evaled


def trainModel(model, x_train, y_train, x_test, y_test):

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score


def createChildPop(parentPop):
    sortedParent = sorted(
        parentPop, key=lambda fitness: fitness[1][1], reverse=True)
    childPop = []
    for i in range(lambda_):
        childPop.append(sortedParent[i][0])
    for i in range(lambda_, len(parentPop)):
        childPop.append([random.sample(kernel_vals), random.sample(range(len(act_list)), random.sample(0, 1), random.sample(range(len(act_list))), random.sample(0, 1), random.sample(range(len(act_list))])
    return childPop

# [kernelsize, 1st activation, 1st dropout, 2nd activation, 2nd dropout, 3rd activation]

# SEARCH SPACE
# setup networks
for i in range(pop_size):
    parent_pop.append([default_kernel, default_act, 0.0, default_act, 0.0, default_act])

# local search through space
for i in range(num_gens):
    parent_pop_evaluated = evaluateNetworks(parent_pop)
    sortedParent = sorted(
        parent_pop, key=lambda fitness: fitness[1][1], reverse=True)
    elites.append((sortedParent[0]), i)
    parent_pop = createChildPop(parent_pop_evaluated)

# print elite from every print_it gen
i = 0
while (i < len(elites)): #TODO:
    print("elite of gen " + str(i) + " with architecture of: in act - " + str(elites[0][i][0][0]) + ", dropout - " + str(
        elites[0][i][0][1]) + ", out act - " + str(elites[0][i][0][2]) + " has accuracy of " + str(elites[0][i][1][1]))
    i += print_rate
sortedElites = sorted(elites, key=lambda score: curr[0][1][1], reverse=True)
print("best network overall is " + str(sortedElites[0][0]) + " with an accuracy of " + str(
    sortedElites[0][1], " and on generation " + str(sortedElites[1])))
