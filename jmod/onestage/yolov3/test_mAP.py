import sys
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
base_path = '/mywork/PGDDS-IIITB/MyPractice'
proj_dir_path = os.path.join(base_path,"MyMScProj")
dataset_base_path = '/mywork/PGDDS-IIITB/MyDatasets'

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sys.path.append(base_path)
sys.path.append(proj_dir_path)

from MyMScProj.jmod.onestage.yolov3.generator_jm import BatchGenerator
from MyMScProj.jmod.preprocess import Preprocess
import numpy as np
import pickle
from tensorflow.keras.models import load_model


saved_model_path=os.path.join(proj_dir_path,'jmod/onestage/yolov3/jm_yolo3_tst6.h5')
annot_csv_file = os.path.join(dataset_base_path,"/MIO-TCD/MIO-TCD-Localization/gt_train.csv")
labels = ['articulated_truck', 'bicycle', 'bus', 'car',
           'motorcycle', 'motorized_vehicle', 'non-motorized_vehicle',
           'pedestrian', 'pickup_truck', 'single_unit_truck', 'work_van']
img_dir = os.path.join(dataset_base_path,"MIO-TCD/MIO-TCD-Localization/train")

import numpy as np

with open(os.path.join(proj_dir_path,"all_insts"), "rb") as rh:
    train_ds = pickle.load(rh)
    rh.close()

with open(os.path.join(proj_dir_path,"seen_labels"), "rb") as rh:
    seen_train_labels = pickle.load(rh)
    rh.close()

infer_model=load_model(saved_model_path, compile=False)  # default is True

anchors=np.array([43,23,
 285,205,
 478,273,
 229,132,
 42,49,
 107,67,
 77,39,
 23,15,
 11,8])
max_grid = [416, 416]
batch_size=32
min_net_size, max_net_size = 288, 448
jitter = 0.1  # how much cropping for image augmentation,
generator_config = {
    'NET_H'         : max_grid[0],
    'NET_W'         : max_grid[1],
    'GRID_H'          : 13,
    'GRID_W'          : 13,
    'LABELS'          : labels,
    'ANCHORS'         : anchors,
    'BATCH_SIZE'      : batch_size,
    'TRUE_BOX_BUFFER' : 32,
    'min_net_size': min_net_size,
    'max_net_size': max_net_size,
    'jitter': jitter
}

def normalize(image):
    return image / 255.

preproc = Preprocess(annot_csv_file, img_dir, labels)
valid_insts, labels = preproc.create_test_instances(train_ds, seen_train_labels, sampling=5000)
validation_batch_generation = BatchGenerator(valid_insts, generator_config, norm=normalize, shuffle=True)

##############################
#  Run the evaluation
##############################
#print("valid_insts", valid_insts)

from MyMScProj.jmod.onestage.yolov3.inference import get_pred_classes_records, annotation_parse, compute_mAP_PascalVOC
pred_classes_records=get_pred_classes_records(infer_model,   validation_batch_generation)
# print("get_pred_classes_records=>", pred_classes_records)

annotation_records, gt_classes_records = annotation_parse(validation_batch_generation)
# print("annotation_records=>", annotation_records)
# print("gt_classes_records=>", gt_classes_records)

iou_threshold=0.05
AP, APs = compute_mAP_PascalVOC(annotation_records, gt_classes_records, pred_classes_records, labels, iou_threshold)
#print("APs", APs)
#print("AP",AP)

####