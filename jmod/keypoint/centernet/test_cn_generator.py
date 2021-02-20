import pickle
import sys
import os
from shutil import rmtree

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
base_path = '/mywork/PGDDS-IIITB/MyPractice'
proj_dir_path = os.path.join(base_path,"MyMScProj")
dataset_base_path = '/mywork/PGDDS-IIITB/MyDatasets'
sys.path.append(base_path)
sys.path.append(proj_dir_path)

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD

from MyMScProj.jmod.keypoint.centernet.generators.miotcd import BatchGenerator
from MyMScProj.jmod.keypoint.centernet.models.resnet import centernet
from MyMScProj.jmod.preprocess import Preprocess

#backend_weight_path=os.path.join(proj_dir_path, 'jmod/keypoint/centernet/ResNet-50-model.keras.h5')
backend_weight_path=os.path.join(proj_dir_path, 'jmod/keypoint/centernet/ResNet-101-model.keras.h5')
tensorboard_dir=os.path.join(proj_dir_path,"tensorboard/centernet_tensorboard_logs_tst6")
saved_weights_dir=os.path.join(proj_dir_path,"jmod/keypoint/centernet/jm_centernet_tst6")
prev_saved_weights_name = "miotcd_01_318.7693_154.9913.h5"
labels = ['articulated_truck', 'bicycle', 'bus', 'car',
           'motorcycle', 'motorized_vehicle', 'non-motorized_vehicle',
           'pedestrian', 'pickup_truck', 'single_unit_truck', 'work_van']
annot_csv_file = os.path.join(dataset_base_path,"/MIO-TCD/MIO-TCD-Localization/gt_train.csv")
train_img_dir = os.path.join(dataset_base_path,"/MIO-TCD/MIO-TCD-Localization/train")

# training params
batch_size=5
input_size=512
freeze_bn=False
epochs=15
steps_per_epoch=10
max_queue_size=8
workers=4
max_objects=20

with open(os.path.join(proj_dir_path,"all_insts"), "rb") as rh:
    train_ds = pickle.load(rh)
    rh.close()

with open(os.path.join(proj_dir_path,"seen_labels"), "rb") as rh:
    seen_train_labels = pickle.load(rh)
    rh.close()

# create training instances
preproc = Preprocess(annot_csv_file, train_img_dir, labels)
print(">>>> Create training instances <<<<")
train_insts, valid_insts, labels, max_box_per_image \
    = preproc.create_training_instances(train_ds, seen_train_labels, sampling=4)


common_args = {
    'batch_size': batch_size,
    'input_size': input_size,
    'labels': labels
    }
train_generator = BatchGenerator(train_insts,base_dir=train_img_dir,**common_args)
inputs, targets = train_generator.__getitem__(index=772)
print("inputs length", len(inputs))
print("batch_images", inputs[0].shape)
print("batch_hms_2", inputs[1].shape)
print("batch_whs", inputs[2].shape)
print("batch_regs", inputs[3].shape)
print("batch_reg_masks", inputs[4].shape)
print("batch_indices", inputs[5].shape)

#num_classes = train_generator.num_classes()

