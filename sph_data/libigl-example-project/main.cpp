#include "main_fairing.cpp"
#include "main_smoothing.cpp"

#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>

#include <iostream>

Eigen::MatrixXd V1, V2;
Eigen::MatrixXi F1, F2;
igl::opengl::glfw::Viewer viewer;

const auto &key_down = [](igl::opengl::glfw::Viewer &viewer,unsigned char key,int mod)->bool
  {
    switch(key)
    {
      case 'r':
      case 'R':
        break;
      default:
        return false;
    }
    // Send new positions, update normals, recenter
    viewer.data().set_vertices(V2);
    viewer.data().compute_normals();
    viewer.core().align_camera_center(V2,F2);
    return true;
  };


int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;

  igl::readOBJ("/home/sontung/work/3d-air-bag-p2/sph_data/mc_solutions/ParticleData_Fluid_100.obj", V1, F1);
  surface_fairing(V1, F1, V2, F2);
  viewer.data().set_mesh(V1, F);
  viewer.callback_key_down = key_down;
  viewer.launch();
}