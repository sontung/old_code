#!/bin/bash

mkdir -p data_heavy

cd libraries/partio/build/Linux-5.4.0-x86_64-optimize/compiled/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
cd ../../../../../
cd ../sph_data/
python main.py

set -euxo pipefail

cd ../preprocess/
python pp_utils.py

cd ../segmentation/
python ear_segment.py

cd ../preprocess/
python main.py

cd ../segmentation/pytorch-deeplab-xception
python inference.py

cd ../../reconstruction/
python solve_airbag.py

cd ../sph_data
python txt2mesh.py

cd libigl-example-project/
mkdir -p build
cd build/
cmake ..
make
./example

cd ../../../reconstruction/
python solve_position.py

cd ../visualization
python main.py

cd ../
python delete_runtime_files.py

