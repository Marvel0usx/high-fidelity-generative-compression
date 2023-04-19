# WiderFace dataset parser.
# Move all images into one folder and aggregate bounding boxes into dictionary.

import os
import glob
import pickle
import scipy.io
import numpy as np


def move(src, dst):
    folders = glob.glob(src + r"/*")
    for subdir in folders:
        os.system("cp " + subdir + r"/*.* " + dst)


def aggregate(train_mat, val_mat, output_dir):
    bbox_dict = dict()

    for path in (train_mat, val_mat):
        mat = scipy.io.loadmat(path)
        parse(mat, bbox_dict)

    with open(output_dir + r"/wider_face_bbox.pickle", "wb") as fout:
        pickle.dump(bbox_dict, fout)


def parse(mat, bbox_dict):
    for i in range(len(mat['event_list'])):
        for j in range(len(mat['file_list'][i,0])):
            file = mat['file_list'][i,0][j,0][0]
            filename = "{}.jpg".format(file)

            # bounding boxes (x,y,w,h)
            bboxs = mat['face_bbx_list'][i,0][j,0]
            # convert from (x, y, w, h) to (x1, y1, x2, y2)
            bboxs[:,2:4] = bboxs[:,2:4] + bboxs[:,0:2]

            bbox_dict[filename] = bboxs.astype(int)
