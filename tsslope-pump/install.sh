#!/bin/bash

CPATH=$(pwd)

export VENV=$CPATH/.pyenv

rm -Rf $VENV

python3.11 -m venv $VENV

source $VENV/bin/activate

pip install -r requirements.txt

pip install -e tsslope-pump-py

echo "import sys;\
      sys.path.append("\"$CPATH"\") "|python

julia -e 'import Pkg; ENV["PYTHON"] = ENV["VENV"]*"/bin/python3"; Pkg.add("PyCall"); Pkg.build("PyCall");'

deactivate

