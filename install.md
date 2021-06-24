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
git submodule update --init --recursive
```

### Step 2: Install environment 
1. Install torch
With cuda 10.1 using torch 1.7
```
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
with cuda 11.1 using torch 1.8
```

```
2. Install detectron2
for torch 1.7 and cuda 10.1
```
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
```
3. Install packages
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

2. Build partio
```
cd libraries/partio
make -j prefix=./compiled install
```

### Step 3: Load checkpoint and prepare data to run
1. Create data_const directory to store checkpoints:
```
mkdir data_const
cd data_consta/
mkdir run
```
You load checkpoint and store in data_const folder. 
2. Prepare data to run:
```
mkdir data_video
cd data_vide/
mkdir all_video
```
copy video folder to all_video.


### Step 4: Running
```
python multiple_video.py
```