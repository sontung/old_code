import os
import sys

import cv2
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
    res_dir = args["dir"]
    sub_folder = os.walk(res_dir).__next__()[1]

    for fld in sub_folder:
        img_list = glob(os.path.join(res_dir, fld) + "/*.png")
        if len(img_list) == 0:
            continue

        img_list = sorted(img_list, key=lambda x: int(x.split("/")[-1].split(".")[0]))

        for path in img_list:
            img = cv2.imread(path)
            name = f"{fld} - {path.split('/')[-1]}"
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.imshow(name, img)
            key = cv2.waitKey(TIME)
            if key == ord("q"):
                print("Quit")
                break

            cv2.destroyAllWindows()

        cv2.waitKey(TIME * 3)

    return


if __name__ == "__main__":
    main()



