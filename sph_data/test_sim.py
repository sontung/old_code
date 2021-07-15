import pysplishsplash as sph

def main():
    base = sph.Exec.SimulatorBase()
    # base.init()
    base.init(sceneFile="/home/sontung/work/3d-air-bag-p2/sph_data/EmitterModel.json")

    gui = sph.GUI.Simulator_GUI_imgui(base)
    base.setGui(gui)
    base.run()

if __name__ == "__main__":
    main()