#!/bin/bash
REBUILD=0
VERBOSE=0
while [ "$1" != "" ]; do
  echo "arg" $1
  case $1 in
    -c)
      shift
      REBUILD=1
      ;;
    -v)
      shift
      VERBOSE=1
      ;;
    *)
      break
  esac
done

if [ -d "build" ] && [ $REBUILD == 1 ]; then
  # if [ -d "build/cuda" ]; then
  #   rm -rf build/cuda
  # fi
  rm -rf build
fi
if [ ! -d "build" ]; then
  mkdir build
  # mkdir build/cuda
fi

cp CMakeLists.txt build/
# cp src/cuda/CMakeLists.txt build/cuda/

cd build
export VERBOSE=$VERBOSE
export WIDELANDS_NINJA_THREADS=20
export TORCH_CUDA_ARCH_LIST="7.0"
export MAX_JOBS=20
cmake .
make -j8
cd ..
python3 install.py

# python3 setup_torch_extension.py clean --all install

