# MachineLearningSlippage

## Overview
Training of Machine Learning Classifiers for Slippage Detection during grasping of objects

## How to use
Initially folders `features` and `plots` are empty and they fill up by usage of components described below.
### Components
**mltraining.ipynb**: ipython notebook (tutorials available online, e.g. http://cs231n.github.io/ipython-tutorial/),
which, if run entirely, downloads all required data first (around 2GB), then computes the features from the input data,
reduces dimensionality, trains several ML classifiers for detecting stable(0) or slip(1) and evaluates their performance.

**mltraining.py**: the python-ic version of the ipython notebook above (**mltraining.ipynb**).

**featext.py**: the module that computes temporal and frequency features, necessary for the above modules.

**__init__.py**: necessary for setting up. Safely ignore!

### Example of running the modules

* **Either run the ipython notebook:**
  ```bash
  ipython notebook
  ```
  Then navigate in the notebook and open **mltraining.ipynb**, from where you are welcome to explore!
* **Or run the python script:**  
  ```bash
  ./mltraining.py
  ```
  And wait for the results, which will require some available RAM (~8GB) and some time, depending on the machine.
