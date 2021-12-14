#!/bin/bash
WIDELANDS_NINJA_THREADS=8 TORCH_CUDA_ARCH_LIST="7.0" MAX_JOBS=8 python3 setup.py clean --all install
