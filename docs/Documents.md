# Introduction

The Airbag is software to compute 3D coordinations of airbags and dummy heads based on videos of crashing car tests.

It provides the following functionalities:

* Calculate 3D coordinations of objects.

* Visualize segmentation.

* Visualize 3D illustration.

# Installation

Please read `install.md` or `install_without_git.md`.

# Understand Calculation Processing

Please read `3D-Reconstruction.pdf`.

# Source Code Explanation

* `data_video`: where you give inputs and receive outputs.

* `data_const`: pre-computed arguments for segmentation.

* `libraries`: libraries needed in code.

* `segmentation` and `segmentation_Swin`: code of segmentation.

* `reconstruction`: perform 3D reconstruction.

* The rest folders: buffer or intermediate computation for 3D reconstruction.
