# MachineLearningSlippage

## Overview
Training of Machine Learning Classifiers for Slippage Detection during grasping of objects

## Dependencies
* This package is written in `python` so you should also have it installed, if not already.
* From python you may need the latest versions of the following packages as well, if not already installed:
  - `numpy`
  - `scipy`
  - `pygame`
  - `matplotlib`
  - `scikit-learn`
  - `nitime`
* You may install the above required packages either by following instructions publicly available online, or trying (with precaution):
  either (preferred)
  ```bash
  sudo apt-get update
  sudo apt-get -y install python-pip
  sudo pip install -U numpy scipy matplotlib scikit-learn
  sudo apt-get install python-pygame python-nitime
  ```
  or
  ```bash
  sudo apt-get update
  sudo easy_install --upgrade numpy scipy pygame matplotlib scikit-learn nitime
  ```

## How to use
### Components
**deploy.py**: a hybrid component  
* composed of  
  - a publisher (talking to `/prob[0/1/2]` ros topic)
  <!-- - a publisher (talking to `/optoforce_0_newf` ros topic): transformation of current measured force based on estimation of normal to contact point. -->
  - a subscriber (listening to `/optoforce_[0/1/2]` ros topic)
* requiring 2 arguments
  1. [offline/online]: define whether to use streaming wrench values, with provided label (offline) or without (online)
  2. [0/1/2]: define optoforce identifier

**pub.py**: a publisher talking to `/optoforce_[0/1/2]` ros topic, requiring 2 arguments, as in **deploy.py**

**sub.py**: a subscriber listening to `/prob[0/1/2]` ros topic and plotting the input

### Build the repository

Say your catkin workspace is `~/catkin_ws`. Then you execute in a terminal the following:
```bash
cd ~/catkin_ws
catkin_make
source ~/catkin_ws/devel/setup.bash
```
*Note that the above procedure of `catkin_make` is not necessary because we are using python.
It is needed when programming in C++ though, so we use it for consistency.*

Additionally you will have to make all 3 python components executable:
```bash
cd ~/catkin_ws/src/slippagedetection/scripts
chmod +x *.py
```

<!-- ### Download required input files

For offline mode the input file is quite big (~200MB) so you have to download it from [here](https://dl.dropboxusercontent.com/u/1047739/testfft_1024_1_lmdb.zip) and
unzip it or equivalently execute in a terminal:
```bash
cd ~/catkin_ws/src/mynet/scripts
wget https://dl.dropboxusercontent.com/u/1047739/testfft_1024_1_lmdb.zip
unzip testfft_1024_1_lmdb.zip
```

If you don't have unzip, just install it by typing in a terminal:
```bash
sudo apt-get install unzip
``` -->

### Example of running the repository

* **Either run one of the launch files provided:**  

  ```bash
  roslaunch slippagedetection deploy.launch
  ```
* **Or run each node separately:**  

1. Initially launch the roscore in a terminal:

  ```bash
  roscore
  ```

2. Subsequently launch in a new terminal the **deploy.py** node, say for offline mode and optoforce_0:

  ```bash
  cd ~/catkin_ws/src/slippagedetection/scripts
  rosrun slippagedetection deploy.py offline 0
  ```

3. Then launch in a new terminal the **sub.py** node:

  ```bash
  rosrun slippagedetection sub.py
  ```

4. Finally launch in a new terminal the **pub.py** node, *with the same mode [offline/online] as* **deploy.py**:

  ```bash
  cd ~/catkin_ws/src/slippagedetection/scripts
  rosrun slippagedetection pub.py offline
  ```
