# MachineLearningSlippage

## Overview

Training of Machine Learning Classifiers for Slippage Detection during grasping of objects.

## How to use

Initially no folders exist, but they are created and filled up by usage of components described below.

## Dataset links

> `Datasets` are automatically downloaded, but links to them along with a detailed explanation are also provided below:

* [`dataset.npz`](https://www.dropbox.com/s/j88wmtx1vvpik1m/dataset.npz?dl=1): force profile for 6 different surface experiments. (numpy zipped file, requires python)

* [`validation.mat`](https://www.dropbox.com/s/r8jl57lij28ljrw/validation.mat?dl=1): force profile for plastic cup experiment. (requires matlab)

> `Dataset format`:

  |  data field  |                        explanation                             |
  |:------------:|:--------------------------------------------------------------:|
  | f1, f2, f    | force `measurements` from finger 1, finger 2, both (f1 and f2) |
  | fd1, fd2, fd | corresponding (per finger) surface `details`                   |
  | l1, l2, l    | corresponding (per finger) `labels`                            |

> `Dataset usage`:

Refer to [`load_data_example.ipynb`](load_data_example.ipynb)

### Requirements

Execution requires ~8GB of RAM and ~5GB of available disc space, among which 1.5GB of datasets and features can be downloaded. It may be required to restart execution if you run out of memory, since several data are saved occasionally and loaded subsequently when needed, sparing useful RAM. Python-ic memory deallocation has been attempted but not successfully achieved, either due to programmer's inefficiency (:sob::trollface:) or python's charming imperfection (:yellow_heart::snake:)..

### Components

**mltraining.ipynb**: ipython notebook (tutorials available online, e.g. [here](http://cs231n.github.io/ipython-tutorial/)),
which, if run entirely, downloads all required data first (around 1.5GB), then computes the features from the input data,
reduces dimensionality, trains several MLP classifiers for detecting stable(0) or slip(1) and evaluates their performance.

**mltraining.py**: the python-ic version of the ipython notebook above (`mltraining.ipynb`).

**featext.py**: the module that computes temporal and frequency features, necessary for the above modules.

**\_\_init\_\_.py**: necessary for setting up. Safely ignore!

### Example of running the modules

* **Either run the ipython notebook:**

  ```bash
  ipython notebook
  ```

  Then navigate in the notebook and open `mltraining.ipynb`, from where you are welcome to explore!

* **Or run the python script:**  

  ```bash
  ./mltraining.py
  ```

  And wait for the results, which will require some available RAM (~8GB) and some time, depending on the machine.
