
cd preprocess/
#python pp_utils.py

cd ../segmentation/
#python ear_segment.py

cd ../preprocess/
#python main.py

cd ../reconstruction/
#python solve_position.py

cd ../sph_data/
python main.py

cd lib-example-project/
mkdir build
cd build/
cmake ..
make
./example

cd segmentation/pytorch-deeplab-xception
python inference.py

cd ../../visualization
python main

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

