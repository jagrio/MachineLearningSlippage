#!/bin/bash

if [[ $UID != 0 ]]; then
    echo "Please run this script with sudo:"
    echo "sudo $0 $*"
    exit 1
fi
# Find current user, not root of course!
user=`who am I | awk '{print $1}'`
# Example
# su somebody <<'EOF'
# command1 -p 'parameter with "quotes" inline'
# command2 -p 'parameter with "quotes" inline'
# EOF
echo "Starting Setup!"
echo "Creating Virtual Environment (virtualenv) to run and install whatever needed!"
if which virtualenv > /dev/null 2>&1;
then
  echo "Virtualenv already installed."
else
  echo "Virtualenv not installed. Installing..."
  apt-get install virtualenv >/dev/null
  echo "Done."
fi
sudo -u $user virtualenv "virtenv"
sudo -u $user source 'virtenv/bin/activate'

echo "Checking for Python version 2.7 ..."
if which python > /dev/null 2>&1;
then
    #Python is installed
    pv=`python --version 2>&1 | awk '{print $2}'`
    if [[ $pv == *"2.7"* ]]
    then
      echo "Python version 2.7 is already installed."
    else
      echo "Python version $pv is installed. Installing 2.7 as well..."
      apt-get install python2.7 >/dev/null
      echo "Done."
    fi
else
    #Python is not installed
    echo "No Python executable is found. Python may not be installed. Installing version 2.7..."
    apt-get install python2.7 >/dev/null
    echo "Done."
fi

# echo "Checking for pip ..."
# if which pip > /dev/null 2>&1;
# then
#     #pip is installed
#     echo "Pip is installed. Updating..."
#     sudo apt-get install --only-upgrade python-pip >/dev/null
#     echo "Done."
# else
#     #pip is not installed
#     echo "Pip is not installed. Installing..."
#     sudo apt-get install python-pip >/dev/null
#     echo "Done."
# fi

echo "Checking for package dependencies and installing whatever needed. This may take a while..."
echo "USER=========================================================="
sudo -u $user pip freeze
echo "ROOT=========================================================="
pip freeze
echo "Checking for setup.py dependencies..."
pip install -U --no-deps ../MLslippage/ #>/dev/null
apt-get update >/dev/null
echo "Checking for pygame..."
tmp=`python -c "import pygame; print pygame.__version__" 2>/dev/null`
echo $tmp
if [ $tmp ] #tmp length is not zero
then
  echo "Pygame is already installed."
else
  echo "Pygame is not installed. Installing..."
  apt-get install python-pygame #>/dev/null
  echo "Done."
fi
echo "Done."
echo "USER=========================================================="
sudo -u $user pip freeze
echo "ROOT=========================================================="
pip freeze
echo "Deactivating Virtualenv"
#sudo -u $user deactivate
echo "Finished Setup!"
