import sys
import numpy as np
import kmeans1d
import pickle
from matplotlib import pyplot as plt
from rec_utils import partition_by_none, grad_diff_compute


def take_care_near_nones(cpd_computations):
    print("taking care near nones")
    best_solution = cpd_computations[:]
    ranges = partition_by_none(cpd_computations)
    start, end = ranges[0]
    path = cpd_computations[start:end]
    grad = np.diff(path)
    if abs(grad[len(path)-2]/grad[len(path)-3]) > 2:
        print(f" detecting unusual gradient of {round(grad[len(path)-2], 2)}"
              f" which is {round(abs(grad[len(path)-2]/grad[len(path)-3]))} times more"
              f" of prev grad {round(grad[len(path)-3], 2)}")
        path[-1] = None
        best_solution[start:end] = path
    if len(ranges) > 1:
        start, end = ranges[1]
        path = cpd_computations[start:end]
        grad1 = path[0]-path[1]
        grad2 = path[1]-path[2]
        if abs(grad1/grad2) > 2:
            print(f" detecting unusual gradient of {round(grad1, 2)}"
                  f" which is {round(abs(grad1/grad2))} times more"
                  f" of prev grad {round(grad2, 2)}")
            path[0] = None
            best_solution[start:end] = path
    return best_solution


def neutralize_head_rot(cpd_computations, head_mask_computations):
    cpd_computations = take_care_near_nones(cpd_computations)
    for idx, val in enumerate(cpd_computations):
        if val is None:
            head_mask_computations[idx] = None
    head_mask_computations = [90-du if du is not None else None for du in head_mask_computations]
    head_mask_computations = look_for_abnormals(head_mask_computations)

    best_solution = cpd_computations[:]
    print("begin to enforce smoothness")
    ranges = partition_by_none(cpd_computations)
    for start, end in ranges:
        cpd_path = cpd_computations[start:end]
        head_path = head_mask_computations[start:end]
        if len(cpd_path) <= 1:
            continue
        best_solution[start:end] = smooth_enforce(cpd_path, head_path)

    plt.plot(cpd_computations, "r")
    plt.plot(best_solution, "b")
    plt.plot(head_mask_computations, "g")
    plt.legend(["cpd", "best", "head mask"])
    plt.savefig("neutral.png")
    plt.close()
    return best_solution


def smooth_enforce(path1, path2):
    """
    select the smoothest solution
    Args:
        path1:
        path2:

    Returns:

    """
    assert None not in path1 and None not in path2
    assert len(path2) == len(path1)
    print(len(path2), len(path1))
    solution1 = [path1[0]]
    solution2 = [path2[0]]

    for x in range(1, len(path1)):
        possible_comp = [path1[x], path2[x], path1[x]-90, path2[x]-90, path1[x]+90, path2[x]+90]
        solution1.append(min(possible_comp, key=lambda du: abs(du-solution1[x-1])))
        solution2.append(min(possible_comp, key=lambda du: abs(du - solution2[x - 1])))
    grad1 = np.sum(np.abs(np.gradient(np.gradient(solution1))))
    grad2 = np.sum(np.abs(np.gradient(np.gradient(solution2))))
    if grad1 < grad2:
        print(f" solution1: using cpd, from {grad1} and {grad2}, we select {grad1}")
        return solution1
    else:
        print(f" solution2: using head mask, from {grad1} and {grad2}, we select {grad2}")
        return solution2


def look_for_abnormals(rot_computation):
    total_grad1 = 0
    total_grad2 = 0
    ori_comp = rot_computation[:]
    ranges = partition_by_none(rot_computation)
    changed = False
    for start, end in ranges:
        if end-start < 2:
            continue
        path = rot_computation[start: end]
        ori_path = path[:]
        total_grad1 += np.sum(np.abs(np.gradient(np.gradient(path))))
        grad2 = np.abs(np.diff(path))
        avg_grad = []
        for idx in range(len(path[:-1])):
            if idx == 0 or path[idx] is None:
                continue
            grad_diff = grad_diff_compute(path, idx)
            if path[idx] != ori_path[idx]:
                print(" changed", idx, path[idx], ori_path[idx], ori_path[idx-1], grad_diff)

            if grad_diff > 2:
                new_path = path[:]
                new_path[idx+1] = path[idx+1]-90
                new_grad_diff = grad_diff_compute(new_path, idx)
                if new_grad_diff < grad_diff:
                    path[idx + 1] = path[idx + 1] - 90
                    changed = True
            avg_grad.append(grad2[idx])
        total_grad2 += np.sum(np.abs(np.gradient(np.gradient(path))))
        rot_computation[start: end] = path
    if changed:
        print(f"reducing grad from {total_grad1} to {total_grad2}")
        plt.plot(ori_comp)
        plt.plot(rot_computation)
        plt.legend(["ori", "new"])
        plt.savefig("abnormal_head.png")
        plt.close()
    return rot_computation


def look_for_abnormals_based_on_ear_sizes(comp, return_selections=False):
    sys.stdin = open("../data_heavy/frames/info.txt")
    lines = [du[:-1] for du in sys.stdin.readlines()]
    all_pixel_dir = "../data_heavy/frames_ear_coord_only"

    ear_sizes = []
    kmeans2frame = {}
    for idx in lines:
        with open("%s/1-%s.png" % (all_pixel_dir, idx), "rb") as fp:
            right_pixels_all = pickle.load(fp)
        if len(right_pixels_all) == 0:
            continue
        else:
            kmeans2frame[int(idx)] = len(ear_sizes)
            ear_sizes.append(len(right_pixels_all))

    res = kmeans1d.cluster(ear_sizes, 2)
    selection = [True for _ in lines]
    if sum(res.clusters)/len(res.clusters) >= 0.5:
        print(f"removing bad ear predictions with ear size diff = {res.centroids[1]/res.centroids[0]}")
        for idx in range(len(comp)):
            if comp[idx] is None or idx+1 not in kmeans2frame:
                continue
            cluster_idx = kmeans2frame[idx+1]
            if res.clusters[cluster_idx] == 0:
                print(" ", idx+1, comp[idx], "removed")
                comp[idx] = None
                selection[idx] = False
    if return_selections:
        return selection
    return comp


def look_for_abnormals_based_on_ear_sizes_tight(comp, return_selections=False):
    sys.stdin = open("../data_heavy/frames/info.txt")
    lines = [du[:-1] for du in sys.stdin.readlines()]
    all_pixel_dir = "../data_heavy/frames_ear_coord_only"

    ear_sizes = []
    kmeans2frame = {}
    for idx in lines:
        with open("%s/1-%s.png" % (all_pixel_dir, idx), "rb") as fp:
            right_pixels_all = pickle.load(fp)
        if len(right_pixels_all) == 0:
            continue
        else:
            kmeans2frame[int(idx)] = len(ear_sizes)
            ear_sizes.append(len(right_pixels_all))

    res = kmeans1d.cluster(ear_sizes, 2)
    selection = [True for _ in lines]
    if sum(res.clusters)/len(res.clusters) >= 0.5 and abs(res.centroids[1]/res.centroids[0]) >= 2:
        print(f"removing bad ear predictions with ear size diff = {res.centroids[1]/res.centroids[0]}")
        for idx in range(len(comp)):
            if comp[idx] is None or idx+1 not in kmeans2frame:
                continue
            cluster_idx = kmeans2frame[idx+1]
            if res.clusters[cluster_idx] == 0:
                print(" ", idx+1, comp[idx], "removed")
                comp[idx] = None
                selection[idx] = False
    if return_selections:
        return selection
    return comp