import numpy as np
np.random.seed(123)  # for reproducibility
from keras.models import Sequential # a linear stack of neural network layers
from keras.layers import Dense, Dropout, Activation, Flatten #core layers
from keras.layers import Convolution2D, MaxPooling2D #CNN layers
from keras.utils import np_utils #utilities
#from matplotlib import pyplot as plt
from keras.datasets import mnist

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape) #(6000,28,28),  60,000 samples in our training set, and the images are 28 pixels x 28 pixels each

#plt.imshow(X_train[0])

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

print(X_train.shape)
# (60000, 1, 28, 28)

#convert our data type to float32 and normalize our data values to the range [0, 1]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print(y_train.shape)


print(y_train[:10])

#splits y_train and y_test data into 10 distinct class labels
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


print(Y_train.shape)
# (60000, 10)

#Actually define the model architecture
model = Sequential()
#define the input layer
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
#first 3 parameters correspond to the number of convolution filters to use,
#the number of rows in each convolution kernel, and the number of columns in each convolution kernel, respectively
print(model.output_shape)

#adding more layers
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25)) #prevents overfitting

#add a fully connected layer and then the output layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



model.fit(X_train, Y_train,
          batch_size=32, nb_epoch=10, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)

print('The score is: ', score)
