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
from tensorflow.keras.models import load_model
from MyMScProj.jmod.keypoint.centernet.models.resnet import centernet
from MyMScProj.jmod.keypoint.centernet.generators.utils import get_affine_transform, affine_transform
from MyMScProj.jmod.keypoint.centernet.utils.image import read_image_bgr, preprocess_image, resize_image

import cv2
import matplotlib.pyplot as plt
import numpy as np

saved_weights_dir=os.path.join(proj_dir_path,"jmod/keypoint/centernet/jm_centernet_tst6")
saved_weights_name = "miotcd_24_8.9395_9.2891.h5"
labels = ['articulated_truck', 'bicycle', 'bus', 'car',
           'motorcycle', 'motorized_vehicle', 'non-motorized_vehicle',
           'pedestrian', 'pickup_truck', 'single_unit_truck', 'work_van']
annot_csv_file = os.path.join(dataset_base_path,"MIO-TCD/MIO-TCD-Localization/gt_train.csv")
test_img_dir = os.path.join(dataset_base_path,"MIO-TCD/MIO-TCD-Localization/test")
saved_model_path = os.path.join(saved_weights_dir, saved_weights_name)

# training params
flip_test = False
nms = True
keep_resolution = False
score_threshold = 0.1
input_w, input_h = 512, 512
model, prediction_model, debug_model, loss_dict = centernet(num_classes=len(labels),
                                                 nms=nms,
                                                 flip_test=flip_test,
                                                 freeze_bn=True,
                                                 score_threshold=score_threshold)

prediction_model.load_weights(saved_model_path, by_name=True, skip_mismatch=True)
prediction_model.summary()

# cn_model=load_model(saved_model_path, compile=True)  # default is True
# print(cn_model.summary())



filename=os.path.join(test_img_dir,'00138488.jpg')
#filename=os.path.join(test_img_dir,'00138455.jpg')
#filename=os.path.join(test_img_dir,'00110652.jpg') # not detecting anything
#filename=os.path.join(test_img_dir,'00110647.jpg')
#filename=os.path.join(test_img_dir, "00111075.jpg")
#image, image_w, image_h = load_image_and_convert_scaled_pixels(filename,input_w, input_h)

# see image before detection
img_plt = cv2.imread(filename)
image_w, image_h, _ = img_plt.shape
plt.subplot(1, 1, 1)
plt.imshow(img_plt)
plt.show()

## preprocess image
def preprocess_image1(image, c, s, tgt_w, tgt_h):
    trans_input = get_affine_transform(c, s, (tgt_w, tgt_h))
    image = cv2.warpAffine(image, trans_input, (tgt_w, tgt_h), flags=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image[..., 0] -= 103.939
    image[..., 1] -= 116.779
    image[..., 2] -= 123.68
    return image

c = np.array([img_plt.shape[1] / 2., img_plt.shape[0] / 2.], dtype=np.float32)
s = max(img_plt.shape[0], img_plt.shape[1]) * 1.0
image = preprocess_image1(img_plt, c, s, input_w, input_h)

#image = preprocess_image(img_plt)
image, scale = resize_image(image)
image.shape
plt.subplot(1, 1, 1)
plt.imshow(image)
plt.show()

# predict
inputs = np.expand_dims(image, axis=0)
centernet_out=prediction_model.predict_on_batch(inputs)[0]

centernet_out.shape


scores = centernet_out[:, 4]
scores

score_threshold = 0.5
indices = np.where(scores > score_threshold)[0]
indices

  # select those detections
detections = centernet_out[indices]
detections_copy = detections.copy()
detections = detections.astype(np.float64)
trans = get_affine_transform(c, s, (input_w // 4, input_h // 4), inv=1)
trans

detections.shape[0]
detections.shape

for j in range(detections.shape[0]):
    detections[j, 0:2] = affine_transform(detections[j, 0:2], trans)
    detections[j, 2:4] = affine_transform(detections[j, 2:4], trans)

detections[0].shape

detections

detections[0, 0:2]


pt = detections[0, 0:2]
t = trans

pt[0]
pt[1]


#np.array([pt[0], pt[1], 1.], dtype=np.float32)  # ValueError: setting an array element with a sequence.
np.array([pt[0], pt[1], 1.], dtype=np.float32).T

np.array([pt[0], pt[1]], dtype=np.float32).T

new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
new_pt = np.dot(t, new_pt )

new_pt
new_pt[:2]

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]



detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, img_plt.shape[1])
detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, img_plt.shape[0])
detections.shape

colors = [np.random.randint(0, 256, 3).tolist() for i in range(len(labels))]
for detection in detections[:25]:
    xmin = int(round(detection[0]))
    ymin = int(round(detection[1]))
    xmax = int(round(detection[2]))
    ymax = int(round(detection[3]))
    score = '{:.4f}'.format(detection[4])
    class_id = int(detection[5])
    color = colors[class_id]
    class_name = labels[class_id]
    label = '-'.join([class_name, score])

    print("xmin {}, ymin {}, xmix {}, ymax {}".format(xmin, ymin, xmax, ymax))
    print("score {}".format(score))
    print("class id {}".format(class_id))
    print("label {}".format(label))
    # ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # cv2.rectangle(img_plt, (xmin, ymin), (xmax, ymax), color, 1)
    # cv2.rectangle(img_plt, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
    # cv2.putText(img_plt, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # plt.subplot(1, 1, 1)
    # plt.imshow(img_plt)
    # plt.show()
#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#cv2.imshow('image', img_plt)















###########
