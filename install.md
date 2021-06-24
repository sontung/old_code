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
1. Install torch with cuda 10.1
```
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
2. Install packages
```
pip install -r requirements.txt
```
2. Install detectron2
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
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
```