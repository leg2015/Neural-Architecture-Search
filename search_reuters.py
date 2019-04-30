'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''
from __future__ import print_function

import random
import numpy as np
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Do we wanna keep these params?
max_words = 1000
batch_size = 32
epochs = 5
pop_size = 5 # change this to 3
lambda_ = 1
sigma = 1.0
num_gens = 20
print_rate = 5

# global lists
act_list = ['tanh', 'softmax', 'elu', 'selu', 'softplus', 'softsign',
            'relu', 'sigmoid', 'hard_sigmoid', 'exponential', 'linear']
default_act = 'linear'

parent_pop = []
parent_pop_evaluated = []
elites = []
totalResults = {}

# Setting up data to be useable by the networks
print('Loading data...')
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                         test_split=0.2)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

num_classes = np.max(y_train) + 1
print(num_classes, 'classes')
print(y_train)

print('Vectorizing sequence data...')
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)






def evaluateNetworks(parentPop):
    evaled = []
    for network in parentPop:
        currModel = makeModel(network)
        modelScore = trainModel(currModel,  x_train, y_train, x_test, y_test)
        evaled.append((network, modelScore))
    return evaled


def createChildPop(parentPop):
    #sortedParent = sorted(parentPop, key=lambda fitness: fitness[1][1], reverse=True)
    childPop = []
    for i in range(lambda_):
        childPop.append(parentPop[i][0])
    for i in range(lambda_, len(parentPop)):
        childPop.append((act_list[random.randint(0,len(act_list)-1)],random.random(),act_list[random.randint(0,len(act_list)-1)])) #makes a random child with one network
        # [1st activation, dropout, 2nd activation]
        #childPop.append([random.sample(range(len(act_list))), random.sample(0, 1), random.sample(range(len(act_list)))])
    return childPop


# toSwap = random.sample(range(len(newGenome)), 2)
#based on reuters_mlp default network
# print('Building model...')
def makeModel(network_params):
    model = Sequential() #want it to be feedforward
    model.add(Dense(512, input_shape=(max_words,))) # input layer stays the same so no data mismatch
    model.add(Activation(network_params[0]))
    model.add(Dropout(network_params[1]))
    model.add(Dense(num_classes)) # creates a layer the size of the possible clasifications
    model.add(Activation(network_params[2]))
    return model


def trainModel(model, x_train, y_train, x_test, y_test):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1)
    score = model.evaluate(x_test, y_test,
                        batch_size=batch_size, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return score


# this model is a feed-forward network with a fully-connected 512 node layer
# setup networks
for i in range(pop_size):
    parent_pop.append([default_act, 0.0, default_act])

# local search through space
for i in range(num_gens):
    parent_pop_evaluated = evaluateNetworks(parent_pop)
    sortedParent = sorted(
        parent_pop_evaluated, key=lambda fitness: fitness[1][1], reverse=True)
    elites.append((sortedParent[0], i))
    parent_pop = createChildPop(parent_pop_evaluated)

# print elite from every print_it gen
sortedElites = sorted(elites, key=lambda score: score[0][1][1], reverse=True)
print('SORTEDELITES:', sortedElites)
champion = sortedElites[0]
bestNetwork = champion[0][0]
print('bestNetwork', bestNetwork)
print('champion',champion)
bestAcc = champion[0][1][1]
chamGen = champion[1]

print("best network overall is ", bestNetwork)
print(" with an accuracy of ",bestAcc)
print(" and on generation ", + chamGen)


# saves results as a csv

download_dir = "reuters_elites_results.csv"  # where you want the file to be downloaded to

csv = open(download_dir, "w")
#"w" indicates that you're writing strings to the file

columnTitleRow = "generation, 1st_act, dropout, 2nd_act, score, accuracy\n"
csv.write(columnTitleRow)

for elite in elites:
    generation = elite[1]
    # network_arch = elite[0][0]
    act_1 = elite[0][0][0]
    dropout = elite[0][0][1]
    act_2 = elite[0][0][2]
    score = elite[0][1][0]
    accuracy = elite[0][1][1]
    row = str(generation) + "," + str(act_1) + "," + str(dropout) + "," + str(act_2) + "," + str(score) + "," + str(accuracy) + "\n"
    csv.write(row)
