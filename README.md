# MyMScProj
YOLOv3 object detection and explanation with RISE and GradCAM

### YOLOv3
This repo has implementation of YOLOv3 in Keras. The code is inspired by below implementations
- https://github.com/david8862/keras-YOLOv3-model-set
- https://github.com/experiencor/keras-yolo3

The software/tools I have used 
- python 3.6
- tensorflow 2.3.1
- keras 2.4.3
- tensorboard 2.3.0

I haven't directly forked these repos rather borrowed few function and modules to aujust my implementation. This is fully custom trained YOLOv3 model with explanation in Grad-CAM and RISE. The ipyhon note for Grad-CAM and RISE is available here
- https://github.com/jmajumde/MyMScProj/blob/main/jmod/onestage/yolov3/yolov3_grad-cam_rise.ipynb

Since, this repo is mainly my masters dissertation from LUMU, there are lots of testing code which I have used to in Atom editor with hydrogen kernel enabled. 

The structure of the repo I kept like below

```
jmod/
├── keypoint
│   ├── centernet
│   │   ├── augmentor
│   │   │   └── __pycache__
│   │   ├── eval
│   │   ├── generators
│   │   │   └── __pycache__
│   │   ├── models
│   │   │   └── __pycache__
│   │   ├── __pycache__
│   │   └── utils
│   │       └── __pycache__
│   └── __pycache__
├── onestage
│   ├── __pycache__
│   └── yolov3
│       ├── models
│       │   └── __pycache__
│       ├── __pycache__
│       └── result
│           └── classes
└── __pycache__
```

Under YOLOv3 I have the complete implementation. The main training module/file is 

- https://github.com/jmajumde/MyMScProj/blob/main/jmod/onestage/yolov3/train.py

The inferensr or evaluation module/file is 

- https://github.com/jmajumde/MyMScProj/blob/main/jmod/onestage/yolov3/test_mAP.py


This implementation is for MIO-TCD dataset available here 
- http://tcd.miovision.com/challenge/dataset/

Since the entire dataset is huge, I have serialized the train set after preprocessing into dictionary format which is available below

- train set instances: https://github.com/jmajumde/MyMScProj/tree/main/dataset_serialized

The dataset can be loaded with below sample python code 

```python
with open(os.path.join(proj_dir_path,"all_insts"), "rb") as rh:
    train_ds = pickle.load(rh)
    rh.close()

with open(os.path.join(proj_dir_path,"seen_labels"), "rb") as rh:
    seen_train_labels = pickle.load(rh)
    rh.close()
```

The train_ds data dictionary then looks like below

```
train_ds[:2]
[
	{'object': [{'name': 'car', 'xmin': 405, 'ymin': 79, 'xmax': 565, 'ymax': 155}, {'name': 'car', 'xmin': 266, 'ymin': 160, 'xmax': 528, 'ymax': 271}], 'filename': '/mywork/PGDDS-IIITB/MyDatasets/MIO-TCD/MIO-TCD-Localization/train/00101947.jpg', 'height': 480, 'width': 720}, 
	{'object': [{'name': 'pickup_truck', 'xmin': 214, 'ymin': 323, 'xmax': 441, 'ymax': 458}, {'name': 'single_unit_truck', 'xmin': 57, 'ymin': 132, 'xmax': 127, 'ymax': 207}], 'filename': '/mywork/PGDDS-IIITB/MyDatasets/MIO-TCD/MIO-TCD-Localization/train/00080403.jpg', 'height': 480, 'width': 720}
]

```

### CenterNet
The model is working but there are still pending work to make the evaluation works


















