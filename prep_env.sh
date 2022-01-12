#!/bin/bash
# use conda, if you don't want to touch shit
# prepare torch related shits, cuda 10.x
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip3 install torch-sparse torch-scatter torch-cluster

# pyg and dgl shits
conda install pyg -c pyg -c conda-forge
conda install -c dglteam dgl-cuda10.1

# prepare some utils
pip3 install opencv-python
pip3 install open3d-python==0.7.0.0
pip3 install scikit-learn
pip3 install tqdm
pip3 install shapely
pip3 install scipy
pip3 install numpy
pip3 install hydra-core
pip3 install omegaconf
pip3 install termcolor

# prepare build things
sudo apt install build-essential libboost-all-dev libgoogle-perftools-dev
sudo apt install cmake

# prepare c++ binds, don't use pip here, as we need some header files
conda install -c conda-forge pybind11
sudo apt install python3-pybind11