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

using namespace Eigen;

void up_sample(MatrixXd originV, MatrixXi originF, int iter, MatrixXd &outV, MatrixXi &outF)
{

    if (iter <= 0)
    {
        outV = originV;
        outF = originF;
    }
    else
    {
        MatrixXd newV;
        MatrixXi newF;
        newV = originV; newF = originF;

        for (int i; i< iter; i++)
        {
            igl::loop(newV, newF, newV, newF);
        }

        outV = newV; outF = newF;
    }
};
