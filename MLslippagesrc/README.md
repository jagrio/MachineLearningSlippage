# MachineLearningSlippage

## Overview

Installing Required Packages and Dependencies

## Dependencies

This package requires:

* `python2.7`, `pip`, `ipython`, `jupyter`, `setuptools`
* From python you may need the latest versions of the following packages as well, if not already installed:
  - `numpy`
  - `scipy` (depending on `gfortran`, `libatlas-base-dev`, `liblapack-dev` as well)
  - `pygame`
  - `matplotlib` (depending on `libpng12-dev`, `libxft-dev `, `libfreetype6-dev` as well)
  - `scikit-learn`
  - `nitime`

## How to setup
You may install the above required packages either by following instructions publicly available online, or trying one of the choices below:

  * The automated way:

    ```bash
    sudo ./setup.sh
    ```
  * The old-school way:
    <!-- - either (preferred) -->

    ```bash
    sudo apt-get update
    sudo apt-get install build-essential python2.7 python2.7-dev \
      python-pip ipython
    sudo apt-get install libfreetype6-dev libxft-dev libpng12-dev \
      gfortran libatlas-base-dev liblapack-dev
    pip install --user -U setuptools
    pip install --user -U jupyter 
    pip install --user --upgrade --no-deps -r requirements.txt
    pip install --user -r requirements.txt
    sudo apt-get install python-pygame
    ```

    <!-- pip install --user -U numpy scipy matplotlib scikit-learn nitime -->

    <!-- - or

    ```bash
    sudo apt-get update
    sudo apt-get install python python-pip ipython jupyter
    sudo apt-get install libfreetype6-dev libxft-dev libpng12-dev \
      gfortran libatlas-base-dev liblapack-dev
    sudo easy_install --upgrade numpy scipy pygame matplotlib scikit-learn nitime
    ``` -->
