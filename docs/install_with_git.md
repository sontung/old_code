### Step 1: Clone code

1. Clone main by https or ssh:
   
   ```
   git clone https://git.hblab.vn/ai/3d-air-bag-p2.git
   ```
   
   ```
   git clone git@git.hblab.vn:ai/3d-air-bag-p2.git
   ```

2. Clone git submodule
   
   ```
   cd 3d-air-bag-p2
   git submodule update --init --recursive
   ```

### Step 2: Install environment

1. Install Torch and Detectron2

With CUDA 10.1

```
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
```

With CUDA 11.1 or 11.0 

```
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch

python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
```

2. Install packages
   
   ```
   sudo apt-get update -y
   sudo apt install swig
   sudo apt-get install -y freeglut3-dev
   sudo apt-get install libssl-dev
   sudo apt-get install xorg-dev libglu1-mesa-dev
   ```

```
pip install -r requirements.txt
```

3. Build partio
   
   ```
   cd libraries/partio
   make -j prefix=./compiled install
   ```

### Step 3: Load checkpoint and prepare data to run

1. Create data_const directory to store checkpoints:
   
   ```
   cd ../..
   mkdir data_const
   cd data_const
   mkdir run
   ```
   
   You load checkpoint and save in data_const folder.

2. Prepare data to run:
   
   ```
   cd ../
   mkdir data_video
   cd data_video/
   mkdir all_video
   ```
   
   copy video folder to all_video.

### Step 4: Running

change directory to 3d-air-bag-p2  

```
cd ../
python multiple_video.py
```

Results are stored in data_video/all_final_vis folder 