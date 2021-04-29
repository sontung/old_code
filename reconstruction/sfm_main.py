import sys
sys.path.append("/home/sontung/tools/OpenSfM")
import opensfm.dataset
import opensfm.actions.detect_features
import opensfm.actions.extract_metadata
import opensfm.actions.match_features
import opensfm.actions.create_tracks
import opensfm.actions.reconstruct
import opensfm.actions.mesh
from utils import complement_point_cloud, visualize_point_cloud
from icecream import install
install()

sfm_dataset = opensfm.dataset.DataSet("data_heavy/sfm_data")

opensfm.actions.extract_metadata.run_dataset(sfm_dataset)
opensfm.actions.detect_features.run_dataset(sfm_dataset)
opensfm.actions.match_features.run_dataset(sfm_dataset)
opensfm.actions.create_tracks.run_dataset(sfm_dataset)

modify_pc = True
if modify_pc:
    name = complement_point_cloud()
else:
    name = None
name = "tracks3.csv"
opensfm.actions.reconstruct.run_dataset(sfm_dataset, name)
# opensfm.actions.mesh.run_dataset(sfm_dataset, name)
visualize_point_cloud()
