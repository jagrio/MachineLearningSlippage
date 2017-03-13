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

echo "Checking for Python version 2.7 ..."
if which python > /dev/null 2>&1;
then
    #Python is installed
    pv=`python --version 2>&1 | awk '{print $2}'`
    if [[ $pv == *"2.7"* ]]
    then
      echo "Python version 2.7 is already installed."
      apt-get install -y python2.7-dev >/dev/null
    else
      echo "Python version $pv is installed. Installing 2.7 as well..."
      apt-get install -y python2.7 python2.7-dev >/dev/null
      echo "Done."
    fi
else
    #Python is not installed
    echo "No Python executable is found. Python may not be installed. Installing version 2.7..."
    apt-get install -y python2.7 python2.7-dev >/dev/null
    echo "Done."
fi

echo "Checking for pip ..."
if which pip > /dev/null 2>&1;
then
    #pip is installed
    echo "Pip is installed. Updating..."
    apt-get install -y --only-upgrade python-pip >/dev/null
    echo "Done."
else
    #pip is not installed
    echo "Pip is not installed. Installing..."
    apt-get install -y python-pip >/dev/null
    echo "Done."
fi

echo "Checking for ipython ..."
if which ipython > /dev/null 2>&1;
then
    #pip is installed
    echo "Ipython is installed. Updating..."
    apt-get install -y --only-upgrade ipython >/dev/null
    echo "Done."
else
    #pip is not installed
    echo "Ipython is not installed. Installing..."
    apt-get install -y ipython >/dev/null
    echo "Done."
fi

# echo "Checking for jupyter ..."
# if which jupyter > /dev/null 2>&1;
# then
#     #pip is installed
#     echo "Jupyter is installed. Updating..."
#     apt-get install -y --only-upgrade jupyter >/dev/null
#     echo "Done."
# else
#     #pip is not installed
#     echo "Jupyter is not installed. Installing..."
#     apt-get install -y jupyter >/dev/null
#     echo "Done."
# fi

echo "Installing or Updating setuptools ..."
pip install -U setuptools >/dev/null
echo "Installing or Updating jupyter ..."
pip install -U jupyter >/dev/null
echo "Installing freetype ... (matplotlib dependency)"
apt-get install -y libfreetype6-dev >/dev/null
echo "Installing png ... (matplotlib dependency)"
apt-get install -y libpng12-dev >/dev/null
echo "Installing fortran, atlas and lapack ... (scipy dependencies)"
apt-get install -y gfortran libatlas-base-dev liblapack-dev >/dev/null

# echo "Creating Virtual Environment (virtualenv) to run and install whatever needed!"
# if which virtualenv > /dev/null 2>&1;
# then
#   echo "Virtualenv already installed."
# else
#   echo "Virtualenv not installed. Installing..."
#   pip install virtualenv >/dev/null
#   echo "Done."
# fi
# virtenv="virtenv"
# sudo -u $user virtualenv $virtenv
# source 'virtenv/bin/activate'

echo "Checking for package dependencies and installing whatever needed. This may take a while..."
#echo "USER============================================================"
#sudo -u $user pip freeze
# echo "VRTENV=========================================================="
# su $user << EOF
# echo $virtenv
# ${virtenv}/bin/pip freeze
# EOF
#sudo -u $user $virtenv/bin/pip freeze
echo "Checking for setup.py dependencies..."
#sudo -u $user
pip install --upgrade --no-deps . >/dev/null
pip install . >/dev/null
# su $user << EOF
# ${virtenv}/bin/pip install -U --no-deps .
# EOF
apt-get update >/dev/null
apt-get remove -y python-matplotlib
echo "Checking for pygame..."
tmp=`python -c "import pygame; print pygame.__version__" 2>/dev/null`
echo $tmp
if [ $tmp ] #tmp length is not zero
then
  echo "Pygame is already installed."
else
  echo "Pygame is not installed. Installing..."
  apt-get install -y python-pygame >/dev/null
  echo "Done."
fi
echo "Done."
#echo "USER============================================================"
#sudo -u $user pip freeze
# echo "VRTENV=========================================================="
# su $user << EOF
# ${virtenv}/bin/pip freeze
# EOF
#sudo -u $user $virtenv/bin/pip freeze
# echo "Deactivating Virtualenv"
#sudo -u $user deactivate
pdir="MLslippage/plots"
fdir="MLslippage/features"
su jagrio << EOF
[ -d $pdir ] || mkdir $pdir
[ -d $fdir ] || mkdir $fdir
EOF
echo "Finished Setup!"
