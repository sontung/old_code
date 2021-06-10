from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import cv2
import sys
import shutil

save_dir = "../data_const/final_vis" 
shutil.rmtree(save_dir, ignore_errors=True)
os.makedirs(save_dir, exist_ok=True)
sys.stdin = open("../data_heavy/frames/info.txt")
lines = [du[:-1] for du in sys.stdin.readlines()]
segment_view = "segment"
recon_view = "recon"
recon_dir = "../data_heavy/saved"
segment_dir = "../data_heavy/frames_seg_abh"
final_im_size = (513, 513)

for idx_recon, idx_seg in enumerate(tqdm(lines, desc="Writing final visualization")):
    seg_im = cv2.imread("%s/1-%s.png" % (segment_dir, idx_seg))
    recon_im = cv2.imread("%s/v1-%s.png" % (recon_dir, idx_recon+1))
    recon_im2 = cv2.imread("%s/v2-%s.png" % (recon_dir, idx_recon+1))

    ori_im = cv2.imread("%s/1-%s.png" % ("../data_heavy/frames", idx_seg))
    ori_im2 = cv2.imread("%s/0-%s.png" % ("../data_heavy/frames", idx_seg))
    if ori_im is None or ori_im2 is None:
        break
    seg_im = cv2.resize(seg_im, final_im_size)
    ori_im = cv2.resize(ori_im, final_im_size)
    blend = cv2.addWeighted(ori_im, 0.3, seg_im, 0.7, 0)
    recon_im = cv2.resize(recon_im, final_im_size)
    recon_im2 = cv2.resize(recon_im2, final_im_size)

    ori_im2 = cv2.resize(ori_im2, final_im_size)
    seg_im2 = cv2.resize(cv2.imread("%s/0-%s.png" % (segment_dir, idx_seg)), final_im_size)
    blend2 = cv2.addWeighted(ori_im2, 0.3, seg_im2, 0.7, 0)
    final_im = np.hstack([blend2, blend])
    final_im2 = np.hstack([recon_im2, recon_im])
    all_im = np.vstack([final_im, final_im2])
    Image.fromarray(all_im).save("%s/%d.png" % (save_dir, idx_recon))
