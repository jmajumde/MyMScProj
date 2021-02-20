import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import pickle

import sys
sys.path.append('/home/jmajumde/PGDDS-IIITB/MyPractice')
sys.path.append('/home/jmajumde/PGDDS-IIITB/MyPractice/MyMScProj')


# custom packages
from jmod.onestage.preprocess import Preprocess
from jmod.onestage.yolov3.bbox import kmeans, BoundingBox
from jmod.onestage.yolov3.inputencoder import ImageEncoder

labels = ['articulated_truck', 'bicycle', 'bus', 'car',
           'motorcycle', 'motorized_vehicle', 'non-motorized_vehicle',
           'pedestrian', 'pickup_truck', 'single_unit_truck', 'work_van']
annot_csv_file = '/media/sf_PGDDS-IIITB-MS-LJMU/MSc/MIO-TCD-Localization/gt_train.csv'
train_img_dir = "/media/sf_PGDDS-IIITB-MS-LJMU/MSc/MIO-TCD-Localization/train"
test_img_dir = "/media/sf_PGDDS-IIITB-MS-LJMU/MSc/MIO-TCD-Localization/test"


eda = Preprocess(annot_csv_file,train_img_dir,labels)

# load the preprocessed instances; to save testing time
def load_all_instances_dict():
    all_insts = []
    seen_labels = {}
    if os.path.exists("/tmp/all_insts"):
        with open("/tmp/all_insts", "rb") as rh:
            train_ds = pickle.load(rh)
            rh.close()

        with open("/tmp/seen_labels", "rb") as rh:
            seen_train_labels = pickle.load(rh)
            rh.close()
    else:
        print("serialized processed dataset file does not exists, preparing afresh....")
        all_insts, seen_labels = eda.prepare_annoted_dict()

        # serialize parsed annoted data set
        with open("/tmp/all_insts", "wb") as wh:
            pickle.dump(all_insts, wh, protocol=pickle.HIGHEST_PROTOCOL)
        wh.close()

        with open("/tmp/seen_labels", "wb") as wh:
            pickle.dump(seen_labels, wh, protocol=pickle.HIGHEST_PROTOCOL)
        wh.close()

    return all_insts, seen_labels

train_ds, seen_train_labels = load_all_instances_dict()
