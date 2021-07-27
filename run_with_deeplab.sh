#!/bin/bash

mkdir -p data_heavy

cd libraries/partio/build/
for partio_dir in */
do
  echo "Found partio in libraries/partio/build/$partio_dir"
  break
done
cd $partio_dir/compiled/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
cd ../../../../../
cd ../sph_data/
python main.py


set -euxo pipefail

cd ../preprocess/
python pp_utils.py

cd ../segmentation/pytorch-deeplab-xception
python inference.py

cd ../
python ear_segment.py

cd ../preprocess/
python main.py

cd ../reconstruction/
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
python solve_position.py -d True

cd ../visualization
python main.py

