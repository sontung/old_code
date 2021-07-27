# Step 1: Download source code

There are two options to download source code.

* The first is for end-user, downloading from our server, is recommended.

* The latter is for developer, cloning from our private GitLab, is not recommended.
1. Our server
   
   We send you the source code as a 3d-air-bag-p2.zip file, please download and extract it with original name.


2. Git cloning 
   
   ```
   git clone https://git.hblab.vn/ai/3d-air-bag-p2.gitClone git submodule
   
   cd 3d-air-bag-p2
   
   git submodule update --init --recursive
   ```

# Step 2: Install Anaconda environment

1. Install Anaconda as the guide on homepage.
   
   ```
   https://docs.anaconda.com/anaconda/install/linux/
   ```

2. Create and activate environment
   
   ```
   conda create -n airbag_phase_2 python=3.8 -y                                                                                           ✔  open-mmlab 🐍 
   
   conda activate airbag_phase_2
   ```
   
   From now on, all command must be run inside `airbag_phase_2` environment.

# Step 3: Install 2D-Segmentation requirements

1. Check CUDA version.
   
   Please make sure the nvidia driver in your computer is available. 
   If not exist, please go to https://www.nvidia.com/Download/index.aspx to download driver and install it. 
   
   ```
   nvidia-smi
   ```
   
   Look at top-right corner for `CUDA Version: xxx`  

2. Install Torch 1.7.1
   
   CUDA 10.1
   
   ```
   pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
   ```
   
   CUDA 10.2
   
   ```
    pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
   ```
   
   CUDA 11.0 or higher
   
   ```
   pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
   ```

3. Install MMCV  
   
   CUDA 10.1
   
   ```
    pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html
   ```
   
   CUDA 10.2
   
   ```
   pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.7.0/index.html
   ```
   
   CUDA 11.0 or higher
   
   ```
   pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
   ```

4. Install MMSegmentation
   
   ```
   cd segmentation_swin && pip install -e . && cd ..
   ```

# Step 4: Install 3D-Reconstruction requirements

1. Install tools for compiling libraries.
   
   ```
   sudo apt-get update -y
   sudo apt install swig
   sudo apt-get install -y freeglut3-dev
   sudo apt-get install libssl-dev
   sudo apt-get install xorg-dev libglu1-mesa-dev
   
   sudo apt remove --purge --auto-remove cmake
   sudo snap install cmake --classic
   ```

2. Build Partio library
   
   ```
   cd libraries/partio && make -j prefix=./compiled install && cd ../..
   ```

3. Install other libraries
   
   ```
   pip install -r requirements.txt
   ```

# Step 5: Load checkpoint

1. Create checkpoints directories
   
   ```
   mkdir -p data_const/run
   mkdir segmentation_swin/checkpoints
   ```

2. Download DeepLab checkpoint
   
   Only admitted Google accounts can download these link.
   
   ```
   https://drive.google.com/file/d/1IHJjiB2n9_WSOqOTGUGHdzIrjcFWNfVO/view?usp=sharing
   ```
   
   Extract then move DeepLab checkpoint into `data_const`.

3. Download Swin-transformer checkpoint
   
   ```
   https://drive.google.com/file/d/1RZQOussrDXTAIwCJ4VsyIBMXKggbwLmZ/view?usp=sharinghttps://drive.google.com/file/d/1RZQOussrDXTAIwCJ4VsyIBMXKggbwLmZ/view?usp=sharing
   ```
   
   Move Swin-transformer checkpoint into `segmentation_swin/checkpoints`

# Step 6: Prepare videos to analyze
Create a directory `data_video` at root directory and subdirectory `all_video` to save all folder videos you want to run:
```
(In the root directory)

mkdir -p data_video
mkdir -p data_video/all_video
```

Each video folder has only 3 videos with SHOULDER view, DRV view, REAR view.  
To get the best results, make sure that the movements in these views are similar or sync.  

Rename 3 required videos to corresponding name:
* SHOULDER view -> `0.mp4` (mp4 is just example).
* DRV view      -> `1.mp4`
* REAR view     -> `2.mp4`

Then, copy video folder to `data_video/all_video`.

# Step 7: Start analyzing

Go back to the root directory and run:

```
(In the root directory)

python multiple_video.py
```
When you run this command, the part of 2d-segmentation using Swin-transformer model.

If you want to use DeepLab model for 2d-segmentation, run:
```
python multiple_video.py -s 1
```

# Step 8: Visualizing

Once completed, results store in `data_video/all_final_vis` as `.png` images. 

There are 2 options to visualize results:

* Just open as normal image.

* Using an app to simulate 3D views with the following program.

```
(In the root directory)

python3 tools/view.py ./data_video/all_final_vis/<<v1>>
```

Replace <<v1>> with the folder of results you want to visualize.

Then, you can press:

* `N` to view next frames.

* `P` to view previous frames.

* `A` to automatically play all the frames.