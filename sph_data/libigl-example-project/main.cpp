#include "main_fairing.cpp"
#include "main_smoothing.cpp"
#include "main_upsample.cpp"

#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>

#include <iostream>
#include <cstdlib>
#include <dirent.h>

using namespace std;


Eigen::MatrixXd V1, V2, V3, V4;
Eigen::MatrixXi F1, F2, F3, F4;
//igl::opengl::glfw::Viewer viewer;

const auto &key_down = [](igl::opengl::glfw::Viewer &viewer,unsigned char key,int mod)->bool
  {
    viewer.data().set_vertices(V2);
    viewer.data().compute_normals();
    viewer.core().align_camera_center(V2,F2);
    return true;
  };

// list all files in a folder
vector<string> list_all_files(string path) {
	DIR* dir;
	struct dirent* ent;

	vector<string> res;
	if ((dir = opendir(path.c_str())) != NULL)
	{
		while ((ent = readdir(dir)) != NULL)
		{
		    string name = ent->d_name;

            if ((name != ".") && (name != ".."))
            {
                res.push_back(ent->d_name);
            }
		}
		closedir(dir);
	}
    else
    {
        cout << "Couldn't open the directory" << endl;
    }
   return res;
};


int main(int argc, char *argv[])
{
  using namespace Eigen;

  vector<string> all_files;

  string obj_folder = "../../mc_solutions/";
  string save_smooth_mesh = "../../mc_solutions_smoothed/";

  all_files = list_all_files(obj_folder);
  float progress = 0.0;
  int files_done = 0;
  int barWidth = 70;
  int nb_files = all_files.size();

  for (auto file: all_files)
  {
      string in_path = obj_folder + file;
      string out_path = save_smooth_mesh + file;

      igl::readOBJ(in_path, V1, F1);
      surface_fairing(V1, F1, V2, F2);
      surface_smoothing(V2, F2, 2, V3, F3);
      up_sample(V3, F3, 2, V4, F4);
      igl::writeOBJ(out_path,V4,F4);

      progress += 1/(float)nb_files;
      files_done+=1;

      cout << "processing mesh with libigl [";
      int pos = barWidth * progress;
      for (int i = 0; i < barWidth; ++i) {
        if (i < pos) cout << "=";
        else if (i == pos) cout << ">";
        else cout << " ";
      }
      cout << "] " <<files_done<<"/"<<nb_files<<"\r";
      cout.flush();

//      viewer.data().set_mesh(V3, F3);
//      viewer.callback_key_down = key_down;
//      viewer.launch();
  }
  cout<<""<<endl;

}