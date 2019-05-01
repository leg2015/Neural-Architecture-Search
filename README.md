# Neural-Architecture-Search
Final project for CSC-424. We are using Keras to generate neural networks that are trained in a supervised fashion on data sets. We will develop a method to explore the space of hyperparameters defining both the network architecture and training procedure to improve performance.


See this page: https://github.com/markdtw/awesome-architecture-search
csv page: http://stanford.edu/~mgorkove/cgi-bin/rpython_tutorials/Writing_Data_to_a_CSV_With_Python.php



got a setup on bryan's gpu rig in the physics lab
installed pip, created a venv to hold keras, tensorflow
source venv
pip install keras , tensorflow-gpu
clone github repo

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
5. Install MatPlot
`pip3 install matplotlib`

### Installing

1. Navigate to the directory you wish to clone the project in a terminal
2. Activate the virtual environment if you set one up (instructions above)
3. Clone the repository using the URL given on [this page](https://github.com/leg2015/Neural-Architecture-Search)
`git clone URL`
## Running the tests

### Reuters Dataset
`python3 search_reuters.py`

### MNIST Dataset
`python3 search_mnist.py`

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* [Lauren Gillespie](https://github.com/leg2015)
* [Elyssa Sliheet](https://github.com/elyssasliheet)
* [Sara Boyd](https://github.com/kayakingCellist)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

