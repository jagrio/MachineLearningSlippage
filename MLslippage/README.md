# MachineLearningSlippage

## Overview

Installing Required Packages and Dependencies

## Dependencies

This package requires:

* `python2.7`, `pip`, `ipython`, `jupyter`, `setuptools`
* From python you may need the latest versions of the following packages as well, if not already installed:
  - `numpy`
  - `scipy`
  - `pygame`
  - `matplotlib`
  - `scikit-learn`
  - `nitime`

## How to setup
You may install the above required packages either by following instructions publicly available online, or trying one of the choices below (with precaution):

  * The automated way:

    ```bash
    sudo ./setup.sh
    ```
  * The old-school way:
    - either (preferred)

    ```bash
    sudo apt-get update
    sudo apt-get install python python-pip ipython jupyter
    sudo pip install -U numpy scipy matplotlib scikit-learn
    sudo apt-get install python-pygame python-nitime
    ```

    - or

    ```bash
    sudo apt-get update
    sudo apt-get install python python-pip ipython jupyter
    sudo easy_install --upgrade numpy scipy pygame matplotlib scikit-learn nitime
    ```
