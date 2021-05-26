
import numpy as np
import cv2
import sys


sys.stdin = open("../data_heavy/frames/info.txt")
lines = [du[:-1] for du in sys.stdin.readlines()]
segment_view = "segment"
recon_view = "recon"
result_dir = "../data_heavy/results"
final_im_size = (480, 640)

for idx in lines:
    seg_im = cv2.imread("%s/segment-1-%s.png" % (result_dir, idx))
    recon_im = cv2.imread("%s/recon-1-%s.png" % (result_dir, idx)) 
    seg_im = cv2.resize(seg_im, final_im_size)
    recon_im = cv2.resize(recon_im, final_im_size)
    final_im = np.hstack([seg_im, recon_im])




