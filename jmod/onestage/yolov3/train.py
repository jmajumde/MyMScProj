import sys
import os

from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import TerminateOnNaN

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
base_path = '/mywork/PGDDS-IIITB/MyPractice'
proj_dir_path = os.path.join(base_path,"MyMScProj")
dataset_base_path = '/mywork/PGDDS-IIITB/MyDatasets'

sys.path.append(base_path)
sys.path.append(proj_dir_path)



from shutil import rmtree
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from MyMScProj.jmod.preprocess import Preprocess
from MyMScProj.jmod.onestage.yolov3.callbacks import CustomTensorBoard, CustomModelCheckpoint
#from MyMScProj.jmod.onestage.yolov3.generator import BatchGenerator
from MyMScProj.jmod.onestage.yolov3.generator_jm import BatchGenerator
from MyMScProj.jmod.onestage.yolov3.models.yolo3_darknet import add_metrics, \
    create_yolo3_model_experiencor
from MyMScProj.jmod.onestage.yolov3.inference import evaluate
import numpy as np
import tensorflow as tf
import pickle


#weight_path='/home/jmajumde/mywork/PGDDS-IIITB/MyPractice/MyMScProj/jmod/onestage/yolov3/yolov3.weights'
weight_path=os.path.join(proj_dir_path, 'jmod/onestage/yolov3/yolov3.weights')
backend_weight_path=os.path.join(proj_dir_path, 'jmod/onestage/yolov3/backend.h5')
anchors=np.array([43,23,
 285,205,
 478,273,
 229,132,
 42,49,
 107,67,
 77,39,
 23,15,
 11,8])
num_anchors=9
num_classes=11
batch_size=32
grid_scales=[1,1,1]
max_grid = [416, 416]
obj_scale = 5
noobj_scale = 1
xywh_scale = 1
class_scale = 1
ignore_thresh = 0.4
warmup_batches = 1
train_times = 2
nb_epochs = 30
steps_per_epoch = 16
freeze_level = -1 # work around freezing/unfreezing layers giving NaN as location loss
min_net_size, max_net_size = 288, 448
jitter = 0.1  # how much cropping for image augmentation,

#tensorboard_dir=os.path.join(proj_dir_path,"tensorboard/yolo3_tensorboard_logs_v2_with5H2epochs16batch")
#tensorboard_dir=os.path.join(proj_dir_path,"tensorboard/yolo3_tensorboard_logs_v1_with15K10epochs50batch")
tensorboard_dir=os.path.join(proj_dir_path,"tensorboard/yolo3_tensorboard_logs_tst6/")
saved_weights_name=os.path.join(proj_dir_path,"jmod/onestage/yolov3/jm_yolo3_tst6.h5")
prev_saved_weights_name=os.path.join(proj_dir_path,"jmod/onestage/yolov3/jm_yolo3_tst5.h5")

labels = ['articulated_truck', 'bicycle', 'bus', 'car',
           'motorcycle', 'motorized_vehicle', 'non-motorized_vehicle',
           'pedestrian', 'pickup_truck', 'single_unit_truck', 'work_van']
annot_csv_file = os.path.join(dataset_base_path,"/MIO-TCD/MIO-TCD-Localization/gt_train.csv")
train_img_dir = os.path.join(dataset_base_path,"/MIO-TCD/MIO-TCD-Localization/train")

def normalize(image):
    return image / 255.

with open(os.path.join(proj_dir_path,"all_insts"), "rb") as rh:
    train_ds = pickle.load(rh)
    rh.close()

with open(os.path.join(proj_dir_path,"seen_labels"), "rb") as rh:
    seen_train_labels = pickle.load(rh)
    rh.close()

def create_callbacks(saved_weights_name, tensorboard_logs, model_to_save):
    if os.path.exists(tensorboard_logs):
        rmtree(tensorboard_logs)
        os.makedirs(tensorboard_logs)

    early_stop = EarlyStopping(
        monitor='loss',
        min_delta=0.01,
        patience=7,
        mode='min',
        verbose=1
    )
    checkpoint = CustomModelCheckpoint(
        model_to_save=model_to_save,
        filepath=saved_weights_name, # + '{epoch:02d}.h5',
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        period=1
    )
    reduce_on_plateau = ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=2,
        verbose=1,
        mode='min',
        epsilon=0.01,
        cooldown=0,
        min_lr=0
    )
    tensorboard = CustomTensorBoard(
        log_dir=tensorboard_logs,
        write_graph=True,
        write_images=True,
        update_freq='batch'
    )
    terminate_on_nan = TerminateOnNaN()
    return [early_stop, checkpoint, reduce_on_plateau, tensorboard, terminate_on_nan]

def dummy_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(y_pred))

# create training instances
preproc = Preprocess(annot_csv_file, train_img_dir, labels)
print(">>>> Create training instances <<<<")
train_insts, valid_insts, labels, max_box_per_image = preproc.create_training_instances(train_ds, seen_train_labels,
                                                                                        sampling=50000)

# generator config
print(">>>> Generate train and validation batches <<<<< ")
generator_config = {
    'NET_H'         : max_grid[0],
    'NET_W'         : max_grid[1],
    'GRID_H'          : 13,
    'GRID_W'          : 13,
    'LABELS'          : labels,
    'ANCHORS'         : anchors,
    'BATCH_SIZE'      : batch_size,
    'TRUE_BOX_BUFFER' : max_box_per_image,
    'min_net_size': min_net_size,
    'max_net_size': max_net_size,
    'jitter': jitter
}

train_batch_generator = BatchGenerator(train_insts, generator_config, norm=normalize, shuffle=True)
validation_batch_generation = BatchGenerator(valid_insts, generator_config, norm=normalize, shuffle=True)
# train_batch_generator = BatchGenerator(
#         instances           = train_insts,
#         anchors             = generator_config['ANCHORS'],
#         labels              = labels,
#         downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
#         max_box_per_image   = max_box_per_image,
#         batch_size          = generator_config['BATCH_SIZE'],
#         min_net_size        = generator_config['min_net_size'],
#         max_net_size        = generator_config['max_net_size'],
#         shuffle             = True,
#         jitter              = generator_config['jitter'],
#         norm                = normalize
#     )
#
# validation_batch_generation = BatchGenerator(
#         instances           = valid_insts,
#         anchors             = generator_config['ANCHORS'],
#         labels              = labels,
#         downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
#         max_box_per_image   = max_box_per_image,
#         batch_size          = generator_config['BATCH_SIZE'],
#         min_net_size        = generator_config['min_net_size'],
#         max_net_size        = generator_config['max_net_size'],
#         shuffle             = True,
#         jitter              = generator_config['jitter'],
#         norm                = normalize
#     )
# create model
print(">>>> Creating yolo3 model <<<<<")
if os.path.exists(prev_saved_weights_name):
    warmup_batches = 0
warmup_batches = warmup_batches * (train_times * len(train_batch_generator))
# train_model, infer_model, loss_dict, darknet, backbone_len = create_yolo3_model(anchors,num_classes,weight_path,max_grid,
#                                                         max_box_per_image,batch_size,ignore_thresh,
#                                                         warmup_batches,grid_scales,obj_scale,
#                                                         noobj_scale,xywh_scale,class_scale)

train_model, infer_model, loss_dict = create_yolo3_model_experiencor(
            nb_class            = num_classes,
            anchors             = anchors,
            max_box_per_image   = max_box_per_image,
            max_grid            = max_grid,
            batch_size          = batch_size,
            warmup_batches      = warmup_batches,
            ignore_thresh       = ignore_thresh,
            grid_scales         = grid_scales,
            obj_scale           = obj_scale,
            noobj_scale         = noobj_scale,
            xywh_scale          = xywh_scale,
            class_scale         = class_scale
        )

# layer_indx=0
# conv_layer_cnt=0
# for layer in darknet.layers:
#     try:
#         if(str(layer.name).startswith('conv2d_')):
#             conv_layer = darknet.get_layer(layer.name)
#             print("conv layer #{}, name {}".format(layer_indx,conv_layer.name))
#             conv_layer_cnt+=1
#         else:
#             print("no convolution #{}, name {}".format(layer_indx,layer.name))
#     except ValueError:
#         print("no convolution #" + str(layer_indx))
#     layer_indx += 1
# print("total layers {}".format(len(darknet.layers)))
# print("Total conv2d layer",conv_layer_cnt)


#freeze the backbone as predefined yolo3 weight is applied to the backbone
# dont freeze last 3 layers which are yolo heads for three different sdcale losses
# if freeze_level in [1,2]:
#     #num = (backbone_len, len(train_model.layers)-3)[freeze_level-1]
#     num = backbone_len
#     for i in range(num):
#         train_model.layers[i].trainable = False
#     print("Freezing the first {} layers of total {} layers".format(num, len(train_model.layers)))
# elif freeze_level == 0:
#     # unfreeze all layers
#     for i in range(len(train_model.layers)):
#         if str(train_model.layers[i].name).startswith('batch_normalization_'):
#             continue
#         train_model.layers[i].trainable = True
#     print("Unfreezing all the layers")

# load backend weights
if os.path.exists(prev_saved_weights_name):
    train_model.load_weights(prev_saved_weights_name)
else:
    train_model.load_weights(backend_weight_path, by_name=True)

# callbacks
callbacks = create_callbacks(saved_weights_name,tensorboard_dir,infer_model)

# add metric
add_metrics(train_model, metric_dict=loss_dict)

# compile model
print(">>>> Compile model <<<<")
optimizer = Adam(lr=1e-4)
#optimizer = Adam(lr=1e-3, clipnorm=0.001)
#optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
#optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0)
train_model.compile(loss=dummy_loss, optimizer=optimizer, run_eagerly=True, metrics=['acc'])

input_tensor = Input(shape=(max_grid[0], max_grid[1], 3), name='image_input')
train_model.build(input_shape=input_tensor)

# fit generator
print(">>>> Train model <<<<")
print(train_model.summary())
# train_model.fit_generator --- gives deprecated warnings hence using model.fit
train_model.fit_generator(generator  = train_batch_generator,
                    steps_per_epoch  = steps_per_epoch,
                    epochs           = nb_epochs + warmup_batches,
                    verbose          = 1,
                    validation_data  = validation_batch_generation,
                    validation_steps = steps_per_epoch//2,
                    callbacks        = callbacks,
                    workers = 4,
                    max_queue_size   = 8)



###############################
#   Run the evaluation
###############################

# compute mAP for all the classes
# average_precisions = evaluate(infer_model, validation_batch_generation)
#
# # print the score
# for label, average_precision in average_precisions.items():
#     print(labels[label] + ': {:.4f}'.format(average_precision))
#     print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))
#


#####

