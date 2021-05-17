
import meshio
data = meshio.read("../data_heavy/sph_solutions/vtk/ParticleData_Fluid_26.vtk")
print(data.points)