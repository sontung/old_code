# PUT VIDEOS INTO RUN FOLDER OF DATA_CONST, AND LABEL 1 FOR DRV VIDEO, LABEL 0 FOR SHOULDER
# DONT PUT ANYTHING ELSE
0 shoulder
1 drv
2 rear

# Introduction

The Airbag is software to compute 3D coordinations of airbags and dummy heads based on videos of crashing car tests.

It provides the following functionalities:

* Calculate 3D coordinations of objects.

* Visualize segmentation.

* Visualize 3D illustration.


# Installation

Please read `docs/install.md`

# Understand Calculation Processing

Please read `docs/3D-Reconstruction.pdf`.

# Source Code Explanation

* `data_video`: where you give inputs and receive outputs.

* `data_const`: pre-computed arguments for segmentation.

* `libraries`: libraries needed in code.

* `segmentation` and `segmentation_Swin`: code of segmentation.

* `reconstruction`: perform 3D reconstruction.

* The rest folders: buffer or intermediate computation for 3D reconstruction.
>>>>>>> f68158c0dd470992a60e055dd89f209bb55ad153
