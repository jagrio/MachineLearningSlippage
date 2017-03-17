#!/bin/bash
#echo $PWD
MY_PATH="`dirname \"$0\"`"
echo "$MY_PATH"
if [[ $UID != 0 ]]; then
  echo "Please run this script with sudo:"
  echo "sudo $0 $* [mode=0]"
  echo "mode: [default] 0 (release) or 1 (debug)"
  exit 1
fi
if [[ $1 == 1 ]]; then
  echo "Debug mode: printing all output!"
  out='1'
else
  echo "Release mode: suppressing most output!"
  exec 3>/dev/null
  out='3'
fi
# Find current user, not root of course!
user=`who am I | awk '{print $1}'`
# Example
# su somebody <<'EOF'
# command1 -p 'parameter with "quotes" inline'
# command2 -p 'parameter with "quotes" inline'
# EOF
echo "Starting Setup!"
echo "Updating Repositories ..."
apt-get update >&$out 2>&1;
echo "Checking for Python version 2.7 ..."
if which python >&$out 2>&1;
then
    #Python is installed
    pv=`python --version 2>&1 | awk '{print $2}'`
    if [[ $pv == *"2.7"* ]]
    then
      echo "Python version 2.7 is already installed."
      apt-get install -y python2.7-dev >&$out
    else
      echo "Python version $pv is installed. Installing 2.7 as well..."
      apt-get install -y python2.7 python2.7-dev >&$out
      echo "Done."
    fi
else
    #Python is not installed
    echo "No Python executable is found. Python may not be installed. Installing version 2.7..."
    apt-get install -y python2.7 python2.7-dev >&$out
    echo "Done."
fi

echo "Checking for pip ..."
if which pip >&$out 2>&1;
then
    #pip is installed
    echo "Pip is installed. Updating..."
    apt-get install -y --only-upgrade python-pip >&$out
    echo "Done."
else
    #pip is not installed
    echo "Pip is not installed. Installing..."
    apt-get install -y python-pip >&$out
    echo "Done."
fi

echo "Checking for ipython ..."
if which ipython >&$out 2>&1;
then
    #pip is installed
    echo "Ipython is installed. Updating..."
    apt-get install -y --only-upgrade ipython >&$out
    echo "Done."
else
    #pip is not installed
    echo "Ipython is not installed. Installing..."
    apt-get install -y ipython >&$out
    echo "Done."
fi

# echo "Checking for jupyter ..."
# if which jupyter > $out 2>&1;
# then
#     #pip is installed
#     echo "Jupyter is installed. Updating..."
#     apt-get install -y --only-upgrade jupyter >&$out
#     echo "Done."
# else
#     #pip is not installed
#     echo "Jupyter is not installed. Installing..."
#     apt-get install -y jupyter >&$out
#     echo "Done."
# fi

echo "Installing or Updating setuptools ..."
pip install -U setuptools >&$out
echo "Installing or Updating jupyter ..."
pip install -U jupyter >&$out
echo "Installing freetype ... (matplotlib dependency)"
apt-get install -y libfreetype6-dev libxft-dev >&$out
echo "Installing png ... (matplotlib dependency)"
apt-get install -y libpng12-dev >&$out
echo "Installing fortran, atlas and lapack ... (scipy dependencies)"
apt-get install -y gfortran libatlas-base-dev liblapack-dev >&$out

# echo "Creating Virtual Environment (virtualenv) to run and install whatever needed!"
# if which virtualenv > $out 2>&1;
# then
#   echo "Virtualenv already installed."
# else
#   echo "Virtualenv not installed. Installing..."
#   pip install virtualenv >&$out
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
#pip install --upgrade --no-deps . >&$out
#pip install . >&$out
pip install --upgrade --no-deps -r $MY_PATH/requirements.txt >&$out
pip install -r $MY_PATH/requirements.txt >&$out
# su $user << EOF
# ${virtenv}/bin/pip install -U --no-deps .
# EOF
apt-get update >&$out
#apt-get remove -y python-matplotlib
echo "Checking for pygame..."
tmp=`python -c "import pygame; print pygame.__version__" 2>&$out`
echo $tmp
if [ $tmp ] #tmp length is not zero
then
  echo "Pygame is already installed."
else
  echo "Pygame is not installed. Installing..."
  apt-get install -y python-pygame >&$out
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
pdir="$MY_PATH/MLslippage/plots"
fdir="$MY_PATH/MLslippage/features"
su $user << EOF
[ -d $pdir ] || mkdir $pdir
[ -d $fdir ] || mkdir $fdir
EOF
echo "Finished Setup!"
