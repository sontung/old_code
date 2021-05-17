import pysplishsplash as sph
import os


def main():
    base = sph.Exec.SimulatorBase()
    output_dir = os.path.abspath("../data_heavy/sph_solutions")

    base.init(sceneFile="/home/sontung/work/3d-air-bag-p2/sph_data/EmitterModel.json", outputDir=output_dir)
    base.setValueFloat(base.STOP_AT, 1.0)  # Important to have the dot to denote a float
    # base.setValueBool(base.VTK_EXPORT, True)
    gui = sph.GUI.Simulator_GUI_imgui(base)
    base.setGui(gui)
    base.run()


if __name__ == "__main__":
    main()