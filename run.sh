cd preprocess/
python pp_utils.py

cd ../segmentation/
python ear_segment.py

cd ../preprocess/
python main.py

cd ../reconstruction/
python solve_position.py
