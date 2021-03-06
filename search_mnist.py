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
pop_size = 3  # change this to 3

lambda_ = 1
sigma = 1.0
num_gens = 5
print_rate = 5

# GLOBAL PARAMETERS
act_list = ['tanh', 'softmax', 'elu', 'selu', 'softplus', 'softsign',
            'relu', 'sigmoid', 'hard_sigmoid', 'exponential', 'linear']
default_act = 'linear'
default_kernel = (2,2)
#kernel_vals = range(1,28/2)
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
    model.add(Flatten())
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

    # print("x test is ", x_test, "and y test is ", y_test)
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
    #sortedParent = sorted(
    #    parentPop, key=lambda fitness: fitness[1][1], reverse=True)
    childPop = []
    for i in range(lambda_):
        childPop.append(parentPop[i][0])
    for i in range(lambda_, len(parentPop)):
        childPop.append((default_kernel, act_list[random.randint(0,len(act_list)-1)], random.random(), act_list[random.randint(0,len(act_list)-1)], random.random(), act_list[random.randint(0,len(act_list)-1)]))
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
        parent_pop_evaluated, key=lambda fitness: fitness[1][1], reverse=True)
    elites.append((sortedParent[0], i))
    parent_pop = createChildPop(parent_pop_evaluated)
    print("generation " + str(i) + " is done") 
# saves results as a csv

# where you want the file to be downloaded to
download_dir = "mnist_elites_results.csv"

csv = open(download_dir, "w")
#"w" indicates that you're writing strings to the file
# [kernelsize, 1st activation, 1st dropout, 2nd activation, 2nd dropout, 3rd activation]

columnTitleRow = "generation, 1st activation, 1st dropout, 2nd activation, 2nd dropout, 3rd activation, score, accuracy\n"
csv.write(columnTitleRow)

for elite in elites:
    generation = elite[1]
    # network_arch = elite[0][0]
    act_1 = elite[0][0][1]
    drop_1 = elite[0][0][2]
    act_2 = elite[0][0][3]
    drop_2 = elite[0][0][4]
    act_3 = elite[0][0][5]
    score = elite[0][1][0]
    accuracy = elite[0][1][1]
    row = str(generation) + "," + str(act_1) + "," + str(drop_1) + \
        "," + str(act_2) + "," + str(drop_2) + "," + str(act_3) + "," + \
        str(score) + "," + str(accuracy) + "\n"
    csv.write(row)


