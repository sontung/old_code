import open3d as o3d

mesh = o3d.io.read_triangle_mesh("/home/sontung/Downloads/lpshead/head.OBJ")
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh)
vis.run()
vis.destroy_window()