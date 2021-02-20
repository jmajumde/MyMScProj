"""
YOLOv3 model in Keras

inspired by https://github.com/david8862/keras-YOLOv3-model-set/blob/33d3355256123b40f49c069958aa0a4fcf526dc7/yolo3/models/yolo3_darknet.py
#           https://github.com/experiencor/keras-yolo3
"""
import struct
import sys
import os
import numpy as np

base_path = '/home/jmajumde/PGDDS-IIITB/MyPractice'
proj_dir_path = os.path.join(base_path,"MyMScProj")
dataset_base_path = '/mywork/PGDDS-IIITB/MyDatasets'

sys.path.append(base_path)
sys.path.append(proj_dir_path)


from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D
from MyMScProj.jmod.onestage.yolov3.models.layers import compose, DarknetConv2D_BN_Leaky, yolo3_predictions, conv_block
from MyMScProj.jmod.onestage.yolov3.loss import YoloLayer
from keras.layers.merge import add, concatenate


class WeightReader:
    def __init__(self, weight_file):
        with open(weight_file, 'rb') as w_f:
            major, = struct.unpack('i', w_f.read(4))
            minor, = struct.unpack('i', w_f.read(4))
            revision, = struct.unpack('i', w_f.read(4))
            if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
                w_f.read(8)
            else:
                w_f.read(4)
            transpose = (major > 1000) or (minor > 1000)
            binary = w_f.read()
        self.offset = 0
        self.all_weights = np.frombuffer(binary, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def load_weights(self, model):
        layer_indx=0
        conv_layer=None
        for layer in model.layers:
            try:
                if (str(layer.name).startswith('conv2d_')):
                    conv_layer = model.get_layer(layer.name)
                    print("loading weights of convolution #" + str(layer_indx))
                    if len(conv_layer.get_weights()) > 1:
                        bias = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                        kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                        kernel = kernel.transpose([2, 3, 1, 0])
                        conv_layer.set_weights([kernel, bias])
                    else:
                        kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                        kernel = kernel.transpose([2, 3, 1, 0])
                        conv_layer.set_weights([kernel])
                else:
                    print("no convolution #" + str(layer_indx))
                #if layer_indx not in [81, 93, 105]:
                if layer_indx not in [90, 150, 197]:
                    if (str(layer.name).startswith('batch_normalization_')):
                        norm_layer = model.get_layer(layer.name)
                        size = np.prod(norm_layer.get_weights()[0].shape)
                        beta = self.read_bytes(size)  # bias
                        gamma = self.read_bytes(size)  # scale
                        mean = self.read_bytes(size)  # mean
                        var = self.read_bytes(size)  # variance
                        weights = norm_layer.set_weights([gamma, beta, mean, var])

            except ValueError:
                print("no convolution #" + str(layer_indx))
            layer_indx+=1

    def reset(self):
        self.offset = 0

def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])
    return x


def darknet53_body(x):
    '''
    Darknet53 body having 52 Convolution2D layers
    Trying to replicate as per authors example https://github.com/experiencor/keras-yolo3
    in a simper modular approach
    layer   filters     kernel-size  output
    conv        32          3x3      256x256
    conv        64          3x3/2    128x128
    1 x conv    32          1x1
        conv    64          3x3
        residual                     128x128
    conv        128         3x3/2    64x64
    2 x conv    64          1x1
        conv    128         3x3
        residual                     64x64
    conv        256         3x3/2    32x32
    8 x conv    128         1x1
        conv    256         3x3
        residual                     16x16
    conv        512         3x3/2    16x16
    8 x conv    256         1x1
        conv    512         3x3
        residual                     16x16
    conv        1024        3x3/2    8x8
    4 x conv    512         1x1
        conv    1024        3x3
        residual                     8x8

    Agvpool                 Global
    FC                      1000
    Softmax

    '''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)

    return x

"""
experiencor model as is
"""
def create_yolo3_model_experiencor(
        nb_class,
        anchors,
        max_box_per_image,
        max_grid,
        batch_size,
        warmup_batches,
        ignore_thresh,
        grid_scales,
        obj_scale,
        noobj_scale,
        xywh_scale,
        class_scale):
    #input_image = Input(shape=(None, None, 3))  # net_h, net_w, 3
    input_image = Input(shape=(max_grid[0], max_grid[1], 3), name='image_input')
    true_boxes = Input(shape=(1, 1, 1, max_box_per_image, 4))

    true_yolo_1 = Input(
        shape=(None, None, len(anchors) // 6, 4 + 1 + nb_class))  # grid_h, grid_w, nb_anchor, 5+nb_class
    true_yolo_2 = Input(
        shape=(None, None, len(anchors) // 6, 4 + 1 + nb_class))  # grid_h, grid_w, nb_anchor, 5+nb_class
    true_yolo_3 = Input(
        shape=(None, None, len(anchors) // 6, 4 + 1 + nb_class))  # grid_h, grid_w, nb_anchor, 5+nb_class

    total_loss_loc = 0
    total_loss_conf = 0
    total_loss_class = 0

    # Layer  0 => 4
    x = conv_block(input_image,
                    [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                     {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                     {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                     {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])

    # Layer  5 => 8
    x = conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                        {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])

    # Layer  9 => 11
    x = conv_block(x, [{'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])

    # Layer 12 => 15
    x = conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                        {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
                        {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])

    # Layer 16 => 36
    for i in range(7):
        x = conv_block(x, [
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16 + i * 3},
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17 + i * 3}])

    skip_36 = x

    # Layer 37 => 40
    x = conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])

    # Layer 41 => 61
    for i in range(7):
        x = conv_block(x, [
            {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41 + i * 3},
            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42 + i * 3}])

    skip_61 = x

    # Layer 62 => 65
    x = conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
                        {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])

    # Layer 66 => 74
    for i in range(3):
        x = conv_block(x, [
            {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66 + i * 3},
            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67 + i * 3}])

    # Layer 75 => 79
    x = conv_block(x, [{'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
                        {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
                        {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}],
                    do_skip=False)

    # Layer 80 => 82
    pred_yolo_1 = conv_block(x, [
        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 80},
        {'filter': (3 * (5 + nb_class)), 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 81}],
                              do_skip=False)
    loss_yolo_1, loss_loc, loss_conf, loss_class  = YoloLayer(anchors[12:],
                            [1 * num for num in max_grid],
                            batch_size,
                            warmup_batches,
                            ignore_thresh,
                            grid_scales[0],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, pred_yolo_1, true_yolo_1, true_boxes])
    total_loss_loc+=loss_loc
    total_loss_conf+=loss_conf
    total_loss_class+=loss_class

    # Layer 83 => 86
    x = conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}],
                    do_skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_61])

    # Layer 87 => 91
    x = conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}],
                    do_skip=False)

    # Layer 92 => 94
    pred_yolo_2 = conv_block(x,
                              [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 92},
                               {'filter': (3 * (5 + nb_class)), 'kernel': 1, 'stride': 1, 'bnorm': False,
                                'leaky': False, 'layer_idx': 93}], do_skip=False)
    loss_yolo_2, loss_loc, loss_conf, loss_class = YoloLayer(anchors[6:12],
                            [2 * num for num in max_grid],
                            batch_size,
                            warmup_batches,
                            ignore_thresh,
                            grid_scales[1],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, pred_yolo_2, true_yolo_2, true_boxes])
    total_loss_loc += loss_loc
    total_loss_conf += loss_conf
    total_loss_class += loss_class

    # Layer 95 => 98
    x = conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 96}],
                    do_skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_36])

    # Layer 99 => 106
    pred_yolo_3 = conv_block(x,
                              [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 99},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True,
                                'layer_idx': 100},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,
                                'layer_idx': 101},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True,
                                'layer_idx': 102},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,
                                'layer_idx': 103},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True,
                                'layer_idx': 104},
                               {'filter': (3 * (5 + nb_class)), 'kernel': 1, 'stride': 1, 'bnorm': False,
                                'leaky': False, 'layer_idx': 105}], do_skip=False)
    loss_yolo_3, loss_loc, loss_conf, loss_class = YoloLayer(anchors[:6],
                            [4 * num for num in max_grid],
                            batch_size,
                            warmup_batches,
                            ignore_thresh,
                            grid_scales[2],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, pred_yolo_3, true_yolo_3, true_boxes])
    total_loss_loc += loss_loc
    total_loss_conf += loss_conf
    total_loss_class += loss_class

    train_model = Model([input_image, true_boxes, true_yolo_1, true_yolo_2, true_yolo_3],
                        [loss_yolo_1, loss_yolo_2, loss_yolo_3])
    infer_model = Model(input_image, [pred_yolo_1, pred_yolo_2, pred_yolo_3])
    loss_dict = {'total_loss_loc': total_loss_loc, 'total_loss_conf': total_loss_conf,
                 'total_loss_class': total_loss_class}

    return [train_model, infer_model, loss_dict]

"""
My custom based on 
 - https://github.com/experiencor/keras-yolo3
and 
 https://github.com/david8862/keras-YOLOv3-model-set
"""
def create_yolo3_model(anchors,num_classes,weight_path,max_grid,max_box_per_image,batch_size,ignore_thresh,
                 warmup_batches,grid_scales,obj_scale,
                 noobj_scale,xywh_scale,class_scale):
    input_tensor = Input(shape=(max_grid[0], max_grid[1], 3), name='image_input')
    true_boxes = Input(shape=(1, 1, 1, max_box_per_image, 4))

    num_feature_layers = len(anchors) // 6  # three different scale feature layers in yolo3
    y_true = [Input(shape=(None, None, len(anchors) // 6, num_classes + 5), name='y_true_{}'.format(l)) for l in
              range(num_feature_layers)]

    total_location_loss = 0
    total_confidence_loss = 0
    total_class_loss = 0

    # define model
    darknet = Model(input_tensor, darknet53_body(input_tensor), name='darknet53_backbone')
    # if weight_path is not None:
    #     weight_reader = WeightReader(weight_path)
    #     weight_reader.load_weights(darknet)
    #     #darknet.load_weights(weight_path, by_name=True)
    #     print('Load weights {}.'.format(weight_path))

    backbone_len = len(darknet.layers)

    """
    define three scale output layers as output (y_pred) heads and concate to the respective resnet backbone
    as shown in darnet53_body API comments
    """
    # f1: 13 x 13 x 1024 -> add_22 (layer name)
    f13 = darknet.output

    # f2: 26 x 26 x 512 -> add_18 (layer name)
    f26 = darknet.layers[152].output

    # f2: 52 x 52 x 256 -> add_10 (layer name)
    f52 = darknet.layers[92].output

    f13_channel_num = 1024
    f26_channel_num = 512
    f52_channel_num = 256
    y_pred1, y_pred2, y_pred3 = yolo3_predictions((f13, f26, f52), (f13_channel_num, f26_channel_num, f52_channel_num),
                                                  num_classes)

    """
    Calculate the losses for each of these three y_preds
    """
    y_pred1_loss, location_loss, conf_loss, class_loss = Yolo3loss(anchors[12:],
                                                                   [1 * num for num in max_grid],
                                                                   batch_size,
                                                                   ignore_thresh,
                                                                   warmup_batches,
                                                                   grid_scales[0],
                                                                   obj_scale,
                                                                   noobj_scale,
                                                                   xywh_scale,
                                                                   class_scale)([input_tensor, y_pred1, y_true[0], true_boxes])
    total_location_loss += location_loss
    total_confidence_loss += conf_loss
    total_class_loss += class_loss

    y_pred2_loss, location_loss, conf_loss, class_loss = Yolo3loss(anchors[6:12],
                                                                   [2 * num for num in max_grid],
                                                                   batch_size,
                                                                   ignore_thresh,
                                                                   warmup_batches,
                                                                   grid_scales[1],
                                                                   obj_scale,
                                                                   noobj_scale,
                                                                   xywh_scale,
                                                                   class_scale)([input_tensor, y_pred2, y_true[1], true_boxes])
    total_location_loss += location_loss
    total_confidence_loss += conf_loss
    total_class_loss += class_loss

    y_pred3_loss, location_loss, conf_loss, class_loss = Yolo3loss(anchors[:6],
                                                                   [4 * num for num in max_grid],
                                                                   batch_size,
                                                                   ignore_thresh,
                                                                   warmup_batches,
                                                                   grid_scales[2],
                                                                   obj_scale,
                                                                   noobj_scale,
                                                                   xywh_scale,
                                                                   class_scale)([input_tensor, y_pred3, y_true[2], true_boxes])
    total_location_loss += location_loss
    total_confidence_loss += conf_loss
    total_class_loss += class_loss

    loss = [y_pred1_loss, y_pred2_loss, y_pred3_loss]
    train_model = Model([darknet.input, true_boxes, y_true[0], y_true[1], y_true[2]], loss)
    infer_model = Model(darknet.input, [y_pred1, y_pred2, y_pred3])

    loss_dict = {'location_loss': total_location_loss, 'confidence_loss': total_confidence_loss,'class_loss': total_class_loss}
    #add_metrics(train_model, loss_dict)

    return( train_model, infer_model , loss_dict, darknet, backbone_len)


def create_yolo3_backbone_with_layeredIndx():
    input_image = Input(shape=(None, None, 3))

    # Layer  0 => 4
    x = conv_block(input_image,
                    [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                     {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                     {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                     {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])

    # Layer  5 => 8
    x = conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                        {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])

    # Layer  9 => 11
    x = conv_block(x, [{'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])

    # Layer 12 => 15
    x = conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                        {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
                        {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])

    # Layer 16 => 36
    for i in range(7):
        x = conv_block(x, [
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16 + i * 3},
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17 + i * 3}])

    skip_36 = x

    # Layer 37 => 40
    x = conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])

    # Layer 41 => 61
    for i in range(7):
        x = conv_block(x, [
            {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41 + i * 3},
            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42 + i * 3}])

    skip_61 = x

    # Layer 62 => 65
    x = conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
                        {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])

    # Layer 66 => 74
    for i in range(3):
        x = conv_block(x, [
            {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66 + i * 3},
            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67 + i * 3}])

    # Layer 75 => 79
    x = conv_block(x, [{'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
                        {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
                        {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}],
                    skip=False)

    # Layer 80 => 82
    yolo_82 = conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 80},
                              {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False,
                               'layer_idx': 81}], skip=False)

    # Layer 83 => 86
    x = conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}],
                    skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_61])

    # Layer 87 => 91
    x = conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}],
                    skip=False)

    # Layer 92 => 94
    yolo_94 = conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 92},
                              {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False,
                               'layer_idx': 93}], skip=False)

    # Layer 95 => 98
    x = conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 96}],
                    skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_36])

    # Layer 99 => 106
    yolo_106 = conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 99},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True,
                                'layer_idx': 100},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,
                                'layer_idx': 101},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True,
                                'layer_idx': 102},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,
                                'layer_idx': 103},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True,
                                'layer_idx': 104},
                               {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False,
                                'layer_idx': 105}], skip=False)

    model = Model(input_image, [yolo_82, yolo_94, yolo_106])
    return model


def add_metrics(model, metric_dict):
    '''
    add metric scalar tensor into model, which could be tracked in training
    log and tensorboard callback
    '''
    for (name, metric) in metric_dict.items():
        # seems add_metric() is newly added in tf.keras. So if you
        # want to customize metrics on raw keras model, just use
        # "metrics_names" and "metrics_tensors" as follow:
        #
        #model.metrics_names.append(name)
        #model.metrics_tensors.append(loss)
        model.add_metric(metric, name=name, aggregation='mean')
        #model.add_metric(metric, name=name)



#####
