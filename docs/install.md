# Step 1: Download source code

1. Download - Recommend.
   
   ```
   https://files.hblab.vn/s/dbLczzTapEPsomz
   ```

2. Git cloning.   
   Only for authorized developers.
   
   ```
   git clone git@git.hblab.vn:ai/3d-air-bag-p2.git && cd 3d-air-bag-p2
   
   git checkout stable
   
   git submodule update --init --recursive
   ```

# Step 2: Install Anaconda environment

1. Install Anaconda as the guide on homepage.
   
   ```
   https://docs.anaconda.com/anaconda/install/linux/
   ```

2. Create and activate environment.
   
   ```
   conda create -n airbag_phase_2 python=3.8 -y   
   
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

2. Install Torch 1.7.1.
   
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

3. Install MMCV.  
   
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

4. Install MMSegmentation.
   
   ```
   (In the root folder)
   cd segmentation_swin && pip install -e . && cd ..
   ```

# Step 4: Install 3D-Reconstruction requirements

1. Install tools for compiling libraries.
   
   ```
   (`sudo` commands will requires password.)
   
   sudo apt-get update -y
   
   sudo apt install swig
   sudo apt-get install -y freeglut3-dev
   sudo apt-get install libssl-dev
   sudo apt-get install xorg-dev libglu1-mesa-dev
   sudo apt-get install libomp-dev
   
   (To ensure we got up-to-date CMake)
   sudo apt remove --purge --auto-remove cmake
   sudo snap install cmake --classic
   ```

2. Build Partio library.
   
   ```
   cd libraries/partio && make -j prefix=./compiled install && cd ../..
   ```

3. Install other libraries.
   
   ```
   pip install -r requirements.txt
   ```

# Step 5: Load checkpoint

1. Create checkpoints directories.
   
   ```
   mkdir -pv data_const/run
   mkdir -v segmentation_swin/checkpoints
   ```

2. Download 3D-Reconstruction checkpoint.
   
   Only admitted Google accounts can download these link.
   
   ```
   https://drive.google.com/file/d/1IHJjiB2n9_WSOqOTGUGHdzIrjcFWNfVO/view?usp=sharing
   ```
   
   Extract then move the 02 files into `data_const`.

3. Download Swin-transformer checkpoint.
   
   ```
   https://drive.google.com/file/d/1RZQOussrDXTAIwCJ4VsyIBMXKggbwLmZ/view?usp=sharing
   ```
   
   Do not extract. Just move the file with extension of`*.pth` into `segmentation_swin/checkpoints`

# Step 6: Prepare videos to analyze

1. Create directories of inputs.
   
   ```
   (In the root directory)
   
   mkdir -pv data_video/all_video
   ```

2. Prepare input videos.
   
   For an crashing test experiment, there are 3 videos of SHOULDER view, DRV view, REAR view.  
   
   Rename 3 required videos to:
   
   * SHOULDER view -> `0.mp4` (mp4 is just example).
   
   * DRV view      -> `1.mp4`
   
   * REAR view     -> `2.mp4`.

3. Copy the folder contains those 3 to `data_video/all_video`.  
   Your folders should look alike:
   
   ```
   data_video
   |---all_video
   |---|---experiment_1
   |---|---|---0.mp4
   |---|---|---1.mp4
   |---|---|---2.mp4
   |---|---experiment_2
   |---|---|---0.mp4
   |---|---|---1.mp4
   |---|---|---2.mp4
   ...
   ```

# Step 7: Start analyzing

1. Activate Conda environment.  
   From now on, every command must be executed in the Conda environment.
   
   ```
   conda activate airbag_phase_2
   ```

2. The main program.
   
   ```
   (In the root directory)
   
   python multiple_video.py -d True
   ```
   
   Above command will use the best model -- Swin-transformer. There is an alternative in case Swin-transformer does not work -- DeepLab. 
   
   ```
   python multiple_video.py -s 1
   ```

# Step 8: Visualizing

Once completed, results store in `data_video/all_final_vis`:  

* As images in `.png`,
* As computed parameters in `.pkl`. 

There are 2 options to visualize results:

* Just open as normal image in `data_video/all_final_vis`.

* Using an app to simulate 3D views with the following program.

```
(In the root directory)

conda activate airbag_phase_2

cd tools

python3 view.py ../data_video/all_final_vis/<<experiment_name>>
```

Replace <<experiment_name>> with the folder of results you want to visualize.

Then, you can press:

* `N` to view next frames.

* `P` to view previous frames.

* `A` to automatically play all the frames.
