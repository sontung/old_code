# PUT VIDEOS INTO RUN FOLDER OF DATA_CONST, AND LABEL 1 FOR DRV VIDEO, LABEL 0 FOR SHOULDER
0 shoulder
1 drv
2 rear

# instruction
Run with python 3.8
```
python multiple_video.py
```
inputs and outputs are saved in `data_video`

### class ID
head 64, 128, 128
airbag 192, 128, 128

### dependencies
```
sudo apt-get update -y
sudo apt install swig
sudo apt-get install -y freeglut3-dev
sudo apt-get install libssl-dev
sudo apt-get install xorg-dev libglu1-mesa-dev
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

```
git submodule update --init --recursive
```

```
pip install open3d
pip install scikit-image
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

#### partio
```
cd libraries/partio
make -j prefix=./compiled install
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
