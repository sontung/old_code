cd preprocess/
python pp_utils.py

cd ../segmentation/
python ear_segment.py

cd ../preprocess/
python main.py

cd ../reconstruction/
python solve_position.py

cd ../sph_data/
mkdir mc_solutions
mkdir mc_solutions_smoothed

cd lib-example-project/
mkdir build
cd build/
cmake ..
make
./example


