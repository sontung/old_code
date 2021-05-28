import os
import numpy as np
import open3d


def convert_array_to_pointclouds(points, normals=None):
    if normals:
        pcd = open3d.geometry.PointCloud(points=open3d.utility.Vector3dVector(points),
                                         normals=open3d.utility.Vector3dVector(normals))
    else:
        pcd = open3d.geometry.PointCloud(points=open3d.utility.Vector3dVector(points))

    pcd.paint_uniform_color([0, 0, 0])
    return pcd


def sample_points_torus(large_radius, small_radius, n_sample=500):
    angle_range = np.linspace(0, 2 * np.pi, endpoint=False, num=n_sample)

    point_cloud = np.zeros((n_sample * n_sample, 3))
    cnt = 0
    for i, theta in enumerate(angle_range):
        for j, phi in enumerate(angle_range):
            x = (large_radius + small_radius * np.cos(theta)) * np.cos(phi)
            y = (large_radius + small_radius * np.cos(theta)) * np.sin(phi)
            z = small_radius * np.sin(theta)

            point_cloud[cnt] = [x, y, z]
            cnt += 1

    return point_cloud


def sample_point_sphere(radius, center=(0, 0, 0), n_sample=300):
    x0, y0, z0 = center

    point_cloud = np.zeros((n_sample * n_sample, 3))
    cnt = 0
    for i, theta in enumerate(np.linspace(0, np.pi, endpoint=True, num=n_sample)):
        for j, phi in enumerate(np.linspace(0, 2 * np.pi, endpoint=False, num=n_sample)):
            x = x0 + radius * np.sin(theta) * np.cos(phi)
            y = y0 + radius * np.sin(theta) * np.sin(phi)
            z = z0 + radius * np.cos(theta)

            point_cloud[cnt] = [x, y, z]
            cnt += 1
    return point_cloud


def create_airbag_pointclouds(large_radius_torus, small_radius_torus, radius_sphere):
    torus_points = sample_points_torus(large_radius_torus, small_radius_torus)
    sphere_points = sample_point_sphere(radius_sphere)

    surface_points = []

    for i in range(torus_points.shape[0]):
        x, y, z = torus_points[i]
        du3 = x * x + y * y + z * z
        du4 = radius_sphere * radius_sphere
        if du3 >= du4:
            surface_points.append(torus_points[i])

    for j in range(sphere_points.shape[0]):
        x, y, z = sphere_points[j]
        du1 = np.square(np.sqrt(x * x + y * y) - large_radius_torus) + z * z
        du2 = small_radius_torus * small_radius_torus
        if du1 < du2:
            continue
        else:
            surface_points.append(sphere_points[j])

    pcd = convert_array_to_pointclouds(np.array(surface_points))
    return pcd


def calculate_normal_point_on_torus(phi, theta):
    tx = -np.sin(phi)
    ty = np.cos(phi)
    tz = 0
    sx = np.cos(phi) * (-np.sin(theta))
    sy = np.sin(phi) * (-np.sin(theta))
    sz = np.cos(theta)

    nx = ty * sz - tz * sy
    ny = tz * sx - tx * sz
    nz = tx * sy - ty * sx

    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= length
    ny /= length
    nz /= length
    return [nx, ny, nz]


def sample_points_torus_with_normal(large_radius, small_radius, n_sample=500):
    angle_range = np.linspace(0, 2 * np.pi, endpoint=False, num=n_sample)

    point_cloud = np.zeros((n_sample * n_sample, 3))
    normals = np.zeros((n_sample * n_sample, 3))
    cnt = 0
    for i, theta in enumerate(angle_range):
        for j, phi in enumerate(angle_range):
            x = (large_radius + small_radius * np.cos(theta)) * np.cos(phi)
            y = (large_radius + small_radius * np.cos(theta)) * np.sin(phi)
            z = small_radius * np.sin(theta)

            point_cloud[cnt] = [x, y, z]
            normals[cnt] = calculate_normal_point_on_torus(phi, theta)
            cnt += 1

    return point_cloud, normals


def sample_point_sphere_with_normal(radius, center=(0, 0, 0), n_sample=100):
    x0, y0, z0 = center

    point_cloud = np.zeros((n_sample * n_sample, 3))
    normals = np.zeros((n_sample * n_sample, 3))
    cnt = 0
    for i, theta in enumerate(np.linspace(0, np.pi, endpoint=True, num=n_sample)):
        for j, phi in enumerate(np.linspace(0, 2 * np.pi, endpoint=False, num=n_sample)):
            x = x0 + radius * np.sin(theta) * np.cos(phi)
            y = y0 + radius * np.sin(theta) * np.sin(phi)
            z = z0 + radius * np.cos(theta)

            point_cloud[cnt] = [x, y, z]
            normals[cnt] = [x / radius, y / radius, z / radius]
            cnt += 1
    return point_cloud, normals


def create_airbag_pointclouds_with_normal(large_radius_torus, small_radius_torus, radius_sphere):
    torus_points, torus_normals = sample_points_torus_with_normal(large_radius_torus, small_radius_torus)
    sphere_points, sphere_normals = sample_point_sphere_with_normal(radius_sphere)

    surface_points = []
    surface_normals = []

    for i in range(torus_points.shape[0]):
        x, y, z = torus_points[i]
        du3 = x * x + y * y + z * z
        du4 = radius_sphere * radius_sphere
        if du3 >= du4:
            surface_points.append(torus_points[i])
            surface_normals.append(torus_normals[i])

    for j in range(sphere_points.shape[0]):
        x, y, z = sphere_points[j]
        du1 = np.square(np.sqrt(x * x + y * y) - large_radius_torus) + z * z
        du2 = small_radius_torus * small_radius_torus
        if du1 < du2:
            continue
        else:
            surface_points.append(sphere_points[j])
            surface_normals.append(sphere_normals[j])

    pcd = convert_array_to_pointclouds(np.array(surface_points), np.array(surface_normals))
    return pcd


if __name__ == '__main__':
    obj_pcd = create_airbag_pointclouds(20, 10, 15)
    open3d.visualization.draw_geometries([obj_pcd])
