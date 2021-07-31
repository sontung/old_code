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


if __name__ == '__main__':
    cpd = [-4.613614350882443, -5.222511494332946, -4.841852811644811, -4.655308401334588, -4.875812312811288, -4.358508950807703, -6.790598987822725, -5.638866652573469, -3.631948276858205, -6.124007019100213, -4.857317400731392, None, -3.771841256166447, -4.715140589597301, -3.8172218763601693, -5.190922938395061, -4.804076020091197, -5.375532666978237, -5.816721426621492, None, -4.692687933721537, None, None, -4.666103092235883, -4.391772510631572, None, -5.303123750309153, -5.036586846200181, None, None, -0.006003166497349984, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 8.262978920774072, None, None, -6.345148871766916, None, None, None, None, 1.976343460265817, None, 7.207148403604803, None, 11.783245029256431, 12.532216600403348, 11.591682225823035, 10.156456875366676, 12.178555853505046, 10.924097138433687, 8.670281204322304, 10.139570351339753, 7.95838833543831, 8.026604765128175, 4.4271257503863595, 7.25551666197505, 4.314942753127316, 3.8594690954420083, None, 2.8859439053426836, 1.2385763852446274, -0.5409188625820438]
    head =[99.60973812505279, 99.60973812505279, 99.31477975428977, 99.60973812505279, 99.60973812505279, 99.60973812505279, 99.31477975428977, 99.90418321297388, 99.90418321297388, 99.60973812505279, 99.90418321297388, 99.60973812505279, 99.60973812505279, 99.31477975428977, 99.31477975428977, 99.31477975428977, 99.31477975428977, 99.31477975428977, 99.31477975428977, 99.31477975428977, 99.60973812505279, 100.00797980144135, 99.11417505479122, 99.85308020417482, 100.72885929801006, 99.60973812505279, 99.71056926601348, 99.16234704572172, 98.13010235415598, 96.81821457165188, 95.61758059012683, 65.7990284088415, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 46.59114027119459, 46.36392753160292, 55.32525821692747, 58.25284954260891, 60.82615407617254, 62.0710212913191, 62.470928127543374, 62.45920263562721, 61.76255446183274, 61.48875370589126, 61.09483682316408, 60.82615407617254, 59.67639313745001, 58.773661532170706, 58.027432240512276, 56.39532105389285, 54.931653319709085, 53.880659150520245, 51.48811511651054, 50.07473037815753, 48.03556912505557, 46.08092418666069, 44.78214637490284, 43.90251394661947, 42.83074241027149, 42.17019240711284, 41.941302425904176, 41.27335507676074, 40.23635830927382, 40.076562623303786, 41.15754687208514, 42.47388308838045, 43.95074405911993, 46.041626676009976, 46.67239436108927]
    neutralize_head_rot(cpd, head)