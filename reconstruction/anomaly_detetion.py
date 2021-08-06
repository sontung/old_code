import sys
import numpy as np
import kmeans1d
import pickle
from matplotlib import pyplot as plt
from rec_utils import partition_by_none, grad_diff_compute


def take_care_near_nones(cpd_computations):
    """
    remove any values which are near None and has big gradient.
    """
    print("taking care near nones")
    best_solution = cpd_computations[:]
    ranges = partition_by_none(cpd_computations)

    # first sub-path
    start, end = ranges[0]
    path = cpd_computations[start:end]
    grad = np.diff(path)
    if abs(grad[len(path)-2]/grad[len(path)-3]) > 2 and len(path) >= 3:
        print(f" detecting unusual gradient of {round(grad[len(path)-2], 2)}"
              f" which is {round(abs(grad[len(path)-2]/grad[len(path)-3]))} times more"
              f" of prev grad {round(grad[len(path)-3], 2)}")
        path[-1] = None  # set unusual large gradient to None
        best_solution[start:end] = path

    # second sub-path
    if len(ranges) > 1:
        start, end = ranges[1]
        path = cpd_computations[start:end]
        if len(path) >= 3:
            grad1 = path[0]-path[1]
            grad2 = path[1]-path[2]
            if abs(grad1/grad2) > 2:
                print(f" detecting unusual gradient of {round(grad1, 2)}"
                      f" which is {round(abs(grad1/grad2))} times more"
                      f" of prev grad {round(grad2, 2)}")
                path[0] = None  # set unusual large gradient to None
                best_solution[start:end] = path
    return best_solution


def neutralize_head_rot(cpd_computations, head_mask_computations):
    """
    finding the best solution between head mask rotation and cpd rotation
    """
    try:
        cpd_computations = take_care_near_nones(cpd_computations)
    except IndexError:
        pass
    for idx, val in enumerate(cpd_computations):
        if val is None:
            head_mask_computations[idx] = None
    head_mask_computations = [90-du if du is not None else None for du in head_mask_computations]
    head_mask_computations = look_for_abnormals(head_mask_computations)

    best_solution = cpd_computations[:]
    print("begin to enforce smoothness")
    ranges = partition_by_none(cpd_computations)
    prev_smoothed_paths = []
    for start, end in ranges:
        print(f" from {start} to {end}")
        cpd_path = cpd_computations[start:end]
        head_path = head_mask_computations[start:end]
        if len(cpd_path) <= 1:
            continue
        if len(prev_smoothed_paths) > 0:
            prev_comp = prev_smoothed_paths[-1]
            if start-prev_comp[0] <= 3:
                best_solution[start:end] = smooth_enforce(cpd_path, head_path, prev_comp[1])
            else:
                best_solution[start:end] = smooth_enforce(cpd_path, head_path)
        else:
            best_solution[start:end] = smooth_enforce(cpd_path, head_path)
        prev_smoothed_paths.append((end, best_solution[end-1]))
        print(best_solution[start:end])

    plt.plot(cpd_computations, "r")
    plt.plot(best_solution, "b")
    plt.plot(head_mask_computations, "g")
    plt.legend(["cpd", "best", "head mask"])
    plt.savefig("neutral.png")
    plt.close()
    return best_solution


def smooth_enforce(path1, path2, prev_comp=None):
    """
    select the smoothest solution
    Args:
        path1: cpd
        path2: head
        prev_comp: if close enough, to perform piecewise smooth
    Returns:

    """
    assert None not in path1 and None not in path2
    assert len(path2) == len(path1)
    if prev_comp is None:
        solution1 = [path1[0]]
        solution2 = [path2[0]]

        for x in range(1, len(path1)):
            possible_comp = [path1[x], path2[x], path1[x]-90, path2[x]-90, path1[x]+90, path2[x]+90]
            solution1.append(min(possible_comp, key=lambda du: abs(du-solution1[x-1])))
            solution2.append(min(possible_comp, key=lambda du: abs(du - solution2[x - 1])))
        grad1 = np.sum(np.abs(np.gradient(solution1)))
        grad2 = np.sum(np.abs(np.gradient(solution2)))
        if grad1 < grad2:
            print(f" solution1: using cpd, from {grad1} and {grad2}, we select {grad1}")
            return solution1
        else:
            print(f" solution2: using head mask, from {grad1} and {grad2}, we select {grad2}")
            return solution2
    else:
        print(f" solution: enforcing piecewise smooth as instructed")
        solution = [min([path1[0], path2[0]], key=lambda du: abs(du-prev_comp))]
        for x in range(1, len(path1)):
            possible_comp = [path1[x], path2[x], path1[x]-90, path2[x]-90, path1[x]+90, path2[x]+90]
            solution.append(min(possible_comp, key=lambda du: abs(du-solution[x-1])))
        return solution


def look_for_abnormals(rot_computation):
    """
    iterate over the input, check if any values has unusual large gradient due to miscalculate the axis of the bounding
    box, then re-compute for the smoother solutions.
    """
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

            # unusual large grad
            if grad_diff > 2:
                new_path = path[:]
                new_path[idx+1] = path[idx+1]-90  # rotate the bounding box axis
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
