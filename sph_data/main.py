import pysplishsplash as sph
import os
import open3d as o3d
from airbag_sampler import AirBagPointCloud


def sample():
    pcd = AirBagPointCloud(45, 35, 40).sample_airbag()
    o3d.io.write_point_cloud("../data_heavy/airbag.pcd", pcd)


def main():
    base = sph.Exec.SimulatorBase()
    output_dir = os.path.abspath("../data_heavy/sph_solutions")

    base.init(sceneFile="/media/hblab/01D5F2DD5173DEA0/AirBag/3d-air-bag-p2/sph_data/EmitterModel_option_1.json", outputDir=output_dir)
    base.setValueFloat(base.STOP_AT, 25)  # Important to have the dot to denote a float
    # base.setValueBool(base.VTK_EXPORT, True)
    base.setValueFloat(base.DATA_EXPORT_FPS, 5)
    gui = sph.GUI.Simulator_GUI_imgui(base)
    base.setGui(gui)
    base.run()


if __name__ == "__main__":
    # sample()
    main()
