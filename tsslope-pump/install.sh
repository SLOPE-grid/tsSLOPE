#!/bin/bash

# Load required modules for Julia HDF5/MPI compatibility
module load gcc/11.2.1 2>/dev/null || module load gcc/11 2>/dev/null
module load hdf5-serial/1.14.0

# Export library paths for GCC 11
export LD_LIBRARY_PATH="/usr/tce/packages/gcc/gcc-11.2.1/lib64:/usr/tce/packages/mvapich2/mvapich2-2.3.7-gcc-11.2.1/lib:$LD_LIBRARY_PATH"

CPATH=$(pwd)

export VENV=$CPATH/.pyenv

rm -Rf $VENV

python -m venv $VENV

source $VENV/bin/activate

pip install -r requirements.txt

pip install -e tsslope-pump-py

echo "import sys;\
      sys.path.append("\"$CPATH"\") "|python

julia -e 'import Pkg; Pkg.activate("/p/lustre1/hiop/project/scidac/env_julia"); ENV["PYTHON"]="/p/lustre1/hiop/project/scidac/venv-scidac/bin/python"; Pkg.add("PyCall"); Pkg.build("PyCall");'

deactivate

echo ""
echo "Installation complete!"
echo "NOTE: Before running driver.jl, make sure to load these modules:"
echo "  module load gcc/11"
echo "  module load hdf5-serial/1.14.0"
echo "  export LD_LIBRARY_PATH=\"/usr/tce/packages/gcc/gcc-11.2.1/lib64:/usr/tce/packages/mvapich2/mvapich2-2.3.7-gcc-11.2.1/lib:\$LD_LIBRARY_PATH\""

