import os
from animation import videofig
import skimage.io
import argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir',
                    default='../data_video/all_final_vis',
                    required=True, type=str,
                    help="Results directory")
parser.add_argument("-t", "--time",
                    default=1000,
                    help="Time of waitKey")

args = vars(parser.parse_args())
TIME = args["time"]


def main():
    """
    escape: skip
    enter: pause
    arrows: next/prev
    """

    res_dir = args["dir"]
    sub_folder = os.walk(res_dir).__next__()[1]

    for count, fld in enumerate(sub_folder):
        img_paths = glob(os.path.join(res_dir, fld) + "/*.png")
        print(count, fld)
        if len(img_paths) == 0:
            continue

        img_paths = sorted(img_paths, key=lambda x: int(x.split("/")[-1].split(".")[0]))
        imgs_list = {}
        for path in img_paths:
            akey = int(path.split("/")[-1].split(".")[0])
            img = skimage.io.imread(path)
            imgs_list[akey] = img

        def redraw_fn(f, axes):
            img = imgs_list[f]
            if not redraw_fn.initialized:
                redraw_fn.im = axes.imshow(img, animated=True)
                redraw_fn.initialized = True
            else:
                redraw_fn.im.set_array(img)

        redraw_fn.initialized = False
        videofig(len(imgs_list), redraw_fn, play_fps=100)

    return


if __name__ == "__main__":
    main()



