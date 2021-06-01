#include <igl/barycenter.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/grad.h>
#include <igl/jet.h>
#include <igl/loop.h>
#include <igl/massmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/readDMAT.h>
#include <igl/readOBJ.h>
#include <igl/repdiag.h>
#include <igl/opengl/glfw/Viewer.h>

#include <iostream>

Eigen::MatrixXd V, newV;
Eigen::MatrixXi F, newF;
Eigen::SparseMatrix<double> L;
igl::opengl::glfw::Viewer viewer;

int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;

  // Load a mesh in OFF format
  igl::readOBJ("/home/sontung/work/3d-air-bag-p2/sph_data/mc_solutions/ParticleData_Fluid_10.obj", V, F);

  const auto &key_down = [](igl::opengl::glfw::Viewer &viewer,unsigned char key,int mod)->bool
  {
    switch(key)
    {
      case 'r':
      case 'R':
        newV = V;
        newF = F;
        break;
      case ' ':
      {
        igl::loop(newV, newF, newV, newF);
        break;
      }
      default:
        return false;
    }
    // Send new positions, update normals, recenter
    viewer.data().set_vertices(newV);
    viewer.data().compute_normals();
    viewer.core().align_camera_center(newV, newF);
    return true;
  };


  // Initialize smoothing with base mesh
  newV = V;
  newF = F;
  viewer.data().set_mesh(newV, newF);
  viewer.callback_key_down = key_down;

  cout<<"Press [space] to smooth."<<endl;;
  cout<<"Press [r] to reset."<<endl;;
  return viewer.launch();
}