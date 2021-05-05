### Running 
```
extract frames (preprocess)
ear segment (segmentation)
edge_detection (preprocess)
prepare_pixels_set (preprocess)
```

```
sudo apt-get install xorg-dev libglu1-mesa-dev
```

```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install kornia
```


### Ceres
```
mkdir ceres-bin
cd ceres-bin
cmake ../ceres-solver-2.0.0
make
make test
sudo make install
```


### openCV
```
# Install minimal prerequisites (Ubuntu 18.04 as reference)
sudo apt update && sudo apt install -y cmake g++ wget unzip
# Download and unpack sources
wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/master.zip
unzip opencv.zip
unzip opencv_contrib.zip
# Create build directory and switch into it
mkdir -p build && cd build
# Configure
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-master/modules ../opencv-master
# Build
cmake --build .
sudo make install
```


### openSFM
```
cd libraries/OpenSfM
python setup.py build
```