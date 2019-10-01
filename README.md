# Neural-Architecture-Search
Neural architecture search is a new, popular technique for automatically discovering successful deep neural architectures for a wide variety of domains. Neural architecture search has been implemented with many different types of search algorithms, including Q-learning and evolutionary strategies. We implemented neural architecture search using a simple mu + lambda evolutionary strategy to optimize accuracy on the popular Reuters and MNIST datasets. We evolve architectures with significantly better performance than the baseline architecture found [here](https://github.com/markdtw/awesome-architecture-search) and comparable performance to hand-optimized architectures of similar size. In future work, we hope to also evolve the layers of the networks along with tuing the hyperparameters.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them
1. Install [Python](https://www.python.org/downloads/) (we have been developing with 3.6.7)
2. Install a virtual environment for the project (optional)
Navigate to the directory you wish to have the copy of the project
```
pip3 install virtualenv
virtualenv venv
source venv/bin/activate (to activate the virtual environment)
deactivate (to exit virtual environment, but don't deactivate before installing steps 3-5)
```
3. Install [Keras](https://keras.io/#installation)
`pip3 install keras`
4. Install [TensorFlow](https://www.tensorflow.org/install)
`pip3 install tensorflow`
5. Install [MatPlotLib](https://matplotlib.org/)
`pip3 install matplotlib`

### Installing

1. Navigate to the directory you wish to clone the project in a terminal
2. Activate the virtual environment if you set one up (instructions above)
3. Clone the repository using the URL given on [this page](https://github.com/leg2015/Neural-Architecture-Search)
`git clone URL`
## Running the tests

### Reuters Dataset
The Reuters dataset is a collection of text newswires, feeds of news and magazine articles, from Reuters in 1987. The dataset has a over 46 topics, or categories, that the articles can be characterized under. More information about the Reuters dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Reuters-21578+Text+Categorization+Collection).

To train the Reuters dataset, run the following command: `python3 search_reuters.py`

We drew inspiration for our network model from [TensorFlow](https://www.tensorflow.org/tutorials/keras/basic_classification) and [Keras](https://keras.io/getting-started/sequential-model-guide/) tutorials.


### MNIST Dataset
The MNIST dataset is a collection of 70,000 grayscale handwritten digits. This is a popular dataset becasue it is the first dataset that a deep neural network was able to perform human levels of accuracy with LeNet. More information regarding MNIST can be found [here](http://yann.lecun.com/exdb/mnist/).

To train the MNIST dataset, run the following command: `python3 search_mnist.py`

## Built With

* [Keras](https://keras.io/)
* [TensorFlow](https://www.tensorflow.org/)

## Authors

* [@leg2015](https://github.com/leg2015)
* [@elyssasliheet](https://github.com/elyssasliheet)
* [@kayakingCellist](https://github.com/kayakingCellist)

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) - see our [LICENSE.md](https://github.com/leg2015/Neural-Architecture-Search/blob/master/LICENSE) for details

## Acknowledgments

Thanks to @markdtw for providing the base code we used to implement the neural networks found [here](https://github.com/markdtw/awesome-architecture-search). Also thanks to @mgorkove for the lovely tutorial on how to write data to a CSV with Python found  [here](http://stanford.edu/~mgorkove/cgi-bin/rpython_tutorials/Writing_Data_to_a_CSV_With_Python.php).
 We would also like to thank our professor, [Dr. Schrum](https://people.southwestern.edu/~schrum2/), for his assistance on the project details. Finally, we'd like to thank [@kwg](https://github.com/kwg) for letting us use (and occasionally break) his linux setup.
