import sys
sys.path.append("../libraries/OpenSfM")
import opensfm.dataset
import opensfm.actions.detect_features
import opensfm.actions.extract_metadata
import opensfm.actions.match_features
import opensfm.actions.create_tracks
import opensfm.actions.reconstruct
import opensfm.actions.mesh
from opensfm import io
from opensfm import reconstruction
import cv2
from rec_utils import complement_point_cloud, visualize_point_cloud, dump_into_tracks_osfm
from icecream import install
install()


def main():
    sys.stdin = open("../data_heavy/frames/info.txt")
    lines = [du[:-1] for du in sys.stdin.readlines()]
    dense_corr_dir = "../data_heavy/matching_solutions"
    images_dir = "../data_heavy/frames_ear_only_nonblack_bg"
    for idx in lines:
        sfm_dataset = opensfm.dataset.DataSet("../data_heavy/sfm_data/%s" % idx)

        opensfm.actions.extract_metadata.run_dataset(sfm_dataset)
        opensfm.actions.detect_features.run_dataset(sfm_dataset)
        opensfm.actions.match_features.run_dataset(sfm_dataset)
        opensfm.actions.create_tracks.run_dataset(sfm_dataset)

        im_names = ["1-%s.png" % idx, "0-%s.png" % idx]
        dump_into_tracks_osfm("%s/dense-corr-%s.txt" % (dense_corr_dir, idx),
                              im_names,
                              [cv2.imread("%s/%s" % (images_dir, im)) for im in im_names],
                              "../data_heavy/sfm_data/%s/tracks.csv" % idx)

        opensfm.actions.reconstruct.run_dataset(sfm_dataset)
        visualize_point_cloud("../data_heavy/sfm_data/%s/reconstruction.json" % idx)
        break


if __name__ == '__main__':
    main()