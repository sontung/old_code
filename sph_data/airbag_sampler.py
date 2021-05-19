import os
import numpy as np
import open3d as o3d


class AirBagPointCloud:

    def __init__(self, R_torus=40, r_torus=35, r_sphere=30):
        self.R_torus = R_torus
        self.r_torus = r_torus
        self.r_sphere = r_sphere

    def sample_airbag(self):
        points_torus = self.sample_torus()
        points_sphere = self.sample_sphere()
        points_1, points_2 = self.take_surface(points_torus, points_sphere)
        # self.visualize_airbag(points_1, points_2)
        pcd = self.array_to_point_cloud(np.vstack([points_1, points_2]))
        print(f'The number of point: {len(pcd.points)}')
        return pcd

    def sample_torus(self, n_sample=500):
        angle_range = np.linspace(0, 2*np.pi, endpoint=False, num=n_sample)

        point_cloud = np.zeros((n_sample*n_sample, 3))
        cnt = 0
        for i, theta in enumerate(angle_range):
            for j, phi in enumerate(angle_range):
                x = (self.R_torus + self.r_torus * np.cos(theta)) * np.cos(phi)
                y = (self.R_torus + self.r_torus * np.cos(theta)) * np.sin(phi)
                z = self.r_torus * np.sin(theta)

                point_cloud[cnt] = [x, y, z]
                cnt += 1

        return point_cloud

    def sample_sphere(self, center=(0, 0, 0), n_sample=500):
        x0, y0, z0 = center
        angle_range = np.linspace(0, 2 * np.pi, endpoint=False, num=n_sample)

        point_cloud = np.zeros((n_sample * n_sample, 3))
        cnt = 0
        for i, theta in enumerate(np.linspace(0, np.pi, endpoint=False, num=n_sample)):
            for j, phi in enumerate(np.linspace(0, 2 * np.pi, endpoint=False, num=n_sample)):
                x = x0 + self.r_sphere * np.sin(theta) * np.cos(phi)
                y = y0 + self.r_sphere * np.sin(theta) * np.sin(phi)
                z = z0 + self.r_sphere * np.cos(theta)

                point_cloud[cnt] = [x, y, z]
                cnt += 1
        return point_cloud

    def take_surface(self, pcd_torus, pcd_sphere):
        t_surface_points = []
        for point in pcd_torus:
            x, y, z = point
            du3 = x * x + y * y + z * z
            du4 = self.r_sphere * self.r_sphere
            if du3 >= du4:
                t_surface_points.append(point)

        s_surface_points = []
        for point in pcd_sphere:
            x, y, z = point
            du1 = np.square(np.sqrt(x * x + y * y) - self.R_torus) + z * z
            du2 = self.r_torus * self.r_torus
            if du1 < du2:
                continue
            else:
                s_surface_points.append(point)
        return np.array(t_surface_points), np.array(s_surface_points)

    @staticmethod
    def visualize_airbag(points1, points2):

        point = np.vstack([points1, points2])
        fake_color1 = np.zeros(points1.shape)
        fake_color2 = np.full(points2.shape, 0.6)
        color = np.vstack([fake_color1, fake_color2])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point)
        pcd.colors = o3d.utility.Vector3dVector(color)

        o3d.visualization.draw_geometries([pcd])
        return

    @staticmethod
    def array_to_point_cloud(points, color=None, visualize=False):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if color:
            fake_color = np.full(points.shape, color)
        else:
            fake_color = np.random.rand(1, 3)

        pcd.colors = o3d.utility.Vector3dVector(fake_color)

        if visualize:
            o3d.visualization.draw_geometries([pcd])
        return pcd


if __name__ == '__main__':
    airbag_obj = AirBagPointCloud(45, 35, 40)
    airbag_obj.sample_airbag()
