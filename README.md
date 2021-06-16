# PUT VIDEOS INTO RUN FOLDER OF DATA_CONST, AND LABEL 1 FOR DRV VIDEO, LABEL 0 FOR SHOULDER
0 shoulder
1 drv
2 rear

### dependencies
```
pip install open3d
pip install scikit-learn
pip install kornia
pip install pycpd
pip install numba
pip install pysplishsplash
pip install microdict
pip install meshio
pip install pykdtree
pip install cmake
```

### Running 
```
extract frames (preprocess)
ear segment (segmentation)
simple_preprocess (preprocess)
edge_detection (preprocess)
prepare_pixels_set (preprocess)
matching_main (reconstruction)
recon_uncalib (reconstruction)
solve_position (reconstruction)
```

### running order
```
extract frames (preprocess)
ear segment (segmentation)
edge_detection() (preprocess)
prepare_pixels_set() (preprocess)
simple_preprocess() (preprocess)
solve_position (reconstruction)
```

```
sudo apt-get update -y
sudo apt install swig
sudo apt-get install -y freeglut3-dev
sudo apt-get install libssl-dev
sudo apt-get install xorg-dev libglu1-mesa-dev
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### partio
```
cd libraries/partio/build/Linux-5.4.0-x86_64-optimize/compiled/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
cd libraries/partio
make -j prefix=./compiled install
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
cmake -DOPENCV_ENABLE_NONFREE=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-master/modules ../opencv-master -DPYTHON_DEFAULT_EXECUTABLE=$(which python) 
# Build
cmake --build .
sudo make install
```


### openSFM
```
cd libraries/OpenSfM
python setup.py build
```



| folder name  | responsible command  | meaning |
|---|---|---|
|  ear_segment |  segmentation/ear_segment.py | pretrained weights for ear model |
|  matching_debugs |   |
| frames  |  preprocess/pp_utils.extract_frame | raw frames extracted from videos |
| matching_solutions  |  reconstruction/matching_main.main | matching pairs between frames |
| sfm_data  |   |
| frames_ear_coord_only  |  segmentation/ear_segment.py | stores the coordinates of pixels within the ears |
| point_cloud_solutions  |   |
| frames_ear_only  |  segmentation/ear_segment.py  | images with only the ears 
| frames_ear_only_nonblack_bg  |  preprocess/main.simple_preprocess  | convert black background to non-black (128, 128, 255)
| refined_pixels  | preprocess/main.prepare_pixels_set  | pixels on the edges (lines and elliptic boundaries)
| frames_ear_only_with_edges  | preprocess/main.prepare_pixels_set  | frames with only the edges of the ears
