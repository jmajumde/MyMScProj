import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import binary_crossentropy
#tf.config.experimental_run_functions_eagerly(True)
from tensorflow.keras import backend as K
tf.config.run_functions_eagerly(True)

def get_cell_grid(GRID_W, GRID_H, BATCH_SIZE):
    '''
    Helper function to assure that the bounding box x and y are in the grid cell scale
    == output ==
    for any i=0,1..,batch size - 1
    output[i,5,3,:,:] = array([[3., 5.],
                               [3., 5.],
                               [3., 5.]], dtype=float32)
    '''
    ## cell_x.shape = (1, 13, 13, 1, 1)
    ## cell_x[:,i,j,:] = [[[j]]]
    cell_x = tf.cast((tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1))),tf.float32)
    ## cell_y.shape = (1, 13, 13, 1, 1)
    ## cell_y[:,i,j,:] = [[[i]]]
    cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))
    ## cell_gird.shape = (16, 13, 13, 5, 2)
    ## for any n, k, i, j
    ##    cell_grid[n, i, j, anchor, k] = j when k = 0
    ## for any n, k, i, j
    ##    cell_grid[n, i, j, anchor, k] = i when k = 1
    cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [BATCH_SIZE, 1, 1, 3, 1])
    return (cell_grid)

cell_x_tst = tf.cast((tf.reshape(tf.tile(tf.range(13), [13]), (1, 13, 13, 1, 1))),tf.float32)
cell_x_tst.shape
cell_y_tst = tf.transpose(cell_x_tst,(0,2,1,3,4))
cell_y_tst.shape
batch_size=10
cell_grid_tst = tf.tile(tf.concat([cell_x_tst,cell_y_tst],-1), [batch_size, 1, 1, 3, 1])
cell_grid_tst.shape

def adjust_scale_prediction(y_pred, cell_grid, ANCHORS):
    """
        Adjust prediction

        == input ==

        y_pred : takes any real values
                 tensor of shape = (N batch, NGrid h, NGrid w, NAnchor, 4 + 1 + N class)

        ANCHORS : list containing width and height specializaiton of anchor box
        == output ==

        pred_box_xy : shape = (N batch, N grid x, N grid y, N anchor, 2), contianing [center_y, center_x] rangining [0,0]x[grid_H-1,grid_W-1]
          pred_box_xy[irow,igrid_h,igrid_w,ianchor,0] =  center_x
          pred_box_xy[irow,igrid_h,igrid_w,ianchor,1] =  center_1

          calculation process:
          tf.sigmoid(y_pred[...,:2]) : takes values between 0 and 1
          tf.sigmoid(y_pred[...,:2]) + cell_grid : takes values between 0 and grid_W - 1 for x coordinate
                                                   takes values between 0 and grid_H - 1 for y coordinate

        pred_Box_wh : shape = (N batch, N grid h, N grid w, N anchor, 2), containing width and height, rangining [0,0]x[grid_H-1,grid_W-1]

        pred_box_conf : shape = (N batch, N grid h, N grid w, N anchor, 1), containing confidence to range between 0 and 1

        pred_box_class : shape = (N batch, N grid h, N grid w, N anchor, N class), containing
    """
    BOX = int(len(ANCHORS) / 2)
    ## cell_grid is of the shape of

    ### adjust x and y
    # the bounding box bx and by are rescaled to range between 0 and 1 for given gird.
    # Since there are BOX x BOX grids, we rescale each bx and by to range between 0 to BOX + 1
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid  # bx, by

    ### adjust w and h
    # exp to make width and height positive
    # rescale each grid to make some anchor "good" at representing certain shape of bounding box
    pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(ANCHORS, [1, 1, 1, BOX, 2])  # bw, bh

    ### adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 4])  # prob bb

    ### adjust class probabilities
    pred_box_class = y_pred[..., 5:]  # prC1, prC2, ..., prC20

    return (pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class)

def adjust_predictions_yolo3(y_pred, cell_grid, grid_h, grid_w):
    pred_box_xy = (cell_grid[:, :grid_h, :grid_w, :, :] + tf.sigmoid(y_pred[..., :2]))  # sigma(t_xy) + c_xy
    pred_box_wh = y_pred[..., 2:4]  # t_wh
    pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[..., 4]), 4)  # adjust confidence
    pred_box_class = y_pred[..., 5:]  # adjust class probabilities

    return (pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class)

def extract_ground_truth(y_true):
    ground_truth_bbox_xy = y_true[..., 0:2]
    ground_truth_bbox_wh = y_true[..., 2:4]
    ground_truth_bbox_conf = tf.expand_dims(y_true[...,4],4)
    ground_truth_bbox_class = tf.argmax(y_true[...,5:], axis=-1)

    return(ground_truth_bbox_xy, ground_truth_bbox_wh, ground_truth_bbox_conf, ground_truth_bbox_class)


def calculate_loss_xywh_yolov2(pred_box_xy, pred_box_wh, true_box_xy,
    true_box_wh, true_box_conf, xywh_scale, anchor_box, net_factor):

    #xywh_mask = tf.expand_dims(true_box_conf, axis=-1) * xywh_scale
    xywh_mask = true_box_conf * xywh_scale
    nb_coord_box = tf.reduce_sum(tf.cast(xywh_mask > 0.0, tf.float32))

    xy_loss = tf.reduce_sum(tf.square(pred_box_xy - true_box_xy) * xywh_mask) / (nb_coord_box + 1e-6)/2.
    wh_loss = tf.reduce_sum(tf.square(pred_box_wh - true_box_wh ) * xywh_mask) / (nb_coord_box + 1e-6)/2.

    return xy_loss, wh_loss


def box_giou(b_true, b_pred):
        """
        Calculate GIoU loss on anchor boxes
        Reference Paper:
            "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression"
            https://arxiv.org/abs/1902.09630
        Parameters
        ----------
        b_true: GT boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        b_pred: predict boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        Returns
        -------
        giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """

        b_true_xy = b_true[..., :2]
        b_true_wh = b_true[..., 2:4]
        b_true_wh_half = b_true_wh / 2.
        b_true_mins = b_true_xy - b_true_wh_half
        b_true_maxes = b_true_xy + b_true_wh_half

        b_pred_xy = b_pred[..., :2]
        b_pred_wh = b_pred[..., 2:4]
        b_pred_wh_half = b_pred_wh / 2.
        b_pred_mins = b_pred_xy - b_pred_wh_half
        b_pred_maxes = b_pred_xy + b_pred_wh_half

        intersect_mins = K.maximum(b_true_mins, b_pred_mins)
        intersect_maxes = K.minimum(b_true_maxes, b_pred_maxes)
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b_true_area = b_true_wh[..., 0] * b_true_wh[..., 1]
        b_pred_area = b_pred_wh[..., 0] * b_pred_wh[..., 1]
        union_area = b_true_area + b_pred_area - intersect_area
        # calculate IoU, add epsilon in denominator to avoid dividing by 0
        iou = intersect_area / (union_area + K.epsilon())

        # get enclosed area
        enclose_mins = K.minimum(b_true_mins, b_pred_mins)
        enclose_maxes = K.maximum(b_true_maxes, b_pred_maxes)
        enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        # calculate GIoU, add epsilon in denominator to avoid dividing by 0
        giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + K.epsilon())
        giou = K.expand_dims(giou, -1)

        return giou

def calculate_loss_xywh_yolov3(pred_box_xy, pred_box_wh, true_box_xy,
                               true_box_wh, true_box_conf, xywh_scale, anchor_box,
                               net_factor,xywh_mask, object_mask, batch_size_f):
    # per experiencor code
    box_loss_scale = tf.exp(true_box_wh) * anchor_box / net_factor
    box_loss_scale = tf.expand_dims(2 - box_loss_scale[..., 0] * box_loss_scale[..., 1], axis=4)
    #print("box_loss_scale shape: ", box_loss_scale.shape)
    #xywh_mask = tf.expand_dims(true_box_conf, axis=-1) * xywh_scale

    xy_delta = (pred_box_xy - true_box_xy)
    #xy_delta = tf.expand_dims(binary_crossentropy(true_box_xy,pred_box_xy,from_logits=True), -1)
    #print("xy_delta shape: ", xy_delta.shape)
    xy_loss = object_mask * xy_delta * box_loss_scale *  xywh_scale
    #xy_loss = tf.reduce_sum(tf.square(xy_loss), list(range(1, 5))) / batch_size_f
    xy_loss = tf.reduce_sum(tf.square(xy_loss), axis=(1,2,3))

    wh_delta = (pred_box_wh - true_box_wh)
    #wh_delta = tf.expand_dims(tf.square(pred_box_wh - true_box_wh),-1)
    wh_loss = object_mask * wh_delta * box_loss_scale *  xywh_scale
    #wh_loss = tf.reduce_sum(wh_loss, list(range(1, 5))) / batch_size_f
    wh_loss = tf.reduce_sum(tf.square(wh_loss), axis=(1,2,3))
    #print("xy_loss {}, wh_loss {} for batch_size {}".format(xy_loss, wh_loss, batch_size_f))
    location_loss = xy_loss + wh_loss

    # xywh_mask = true_box_conf * xywh_scale
    # nb_coord_box = tf.reduce_sum(tf.cast(xywh_mask > 0.0, tf.float32))
    # b_true = K.concatenate([true_box_xy, true_box_wh])
    # b_pred = K.concatenate([pred_box_xy, pred_box_wh])
    # giou = box_giou(b_true, b_pred)
    # giou_loss = object_mask * box_loss_scale * (1 - giou)
    # #giou_loss = K.sum(giou_loss) / (nb_coord_box + 1e-6)/2.
    # giou_loss = K.sum(giou_loss)
    # print("giou_loss {} for batch_size {}".format(giou_loss,batch_size_f))
    # giou_loss = giou_loss / batch_size_f
    # location_loss = giou_loss

    return (location_loss)


def calculate_loss_class_yolov2(true_box_conf, true_box_class, pred_box_class, CLASS_SCALE):
    '''
    == output ==
    class_mask
    if object exists in this (grid_cell, anchor) pair and the class object receive nonzero weight
        class_mask[iframe,igridy,igridx,ianchor] = 1
    else:
        0
    '''
    class_mask   = true_box_conf  * CLASS_SCALE ## L_{i,j}^obj * lambda_class

    nb_class_box = tf.reduce_sum(tf.cast(class_mask > 0.0, tf.float32))
    loss_class   = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = true_box_class,
                                                                  logits = pred_box_class)
    loss_class   = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
    return(loss_class)


def calculate_loss_class_yolov3(object_mask, true_box_class, pred_box_class,
                                class_scale, batch_size_f):

    object_mask = tf.cast(object_mask, tf.float32)
    var = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
    class_delta = object_mask * tf.expand_dims(var, axis=4) * class_scale
    #class_loss = tf.reduce_sum(tf.square(class_delta), list(range(1, 5))) / batch_size_f
    class_loss = tf.reduce_sum(class_delta, axis=(1,2,3))
    return class_loss

def calc_loss_conf_yolov3(object_mask, true_box_conf, pred_box_conf, conf_delta,
                          obj_scale, noobj_scale, batch_size_f):

    #conf_delta = object_mask * (pred_box_conf - true_box_conf) * obj_scale + (1 - object_mask) * conf_delta * noobj_scale
    #loss_conf = tf.reduce_sum(tf.square(conf_delta), list(range(1, 5)))

    #obj_loss = (pred_box_conf - true_box_conf)
    obj_loss = tf.expand_dims(binary_crossentropy(true_box_conf, pred_box_conf), -1)
    obj_loss = object_mask * obj_loss * obj_scale + \
               (1 - object_mask) * conf_delta * obj_loss * noobj_scale
    #loss_conf = tf.reduce_sum(tf.square(obj_loss), list(range(1,5))) / batch_size_f
    loss_conf = tf.reduce_sum(obj_loss, axis=(1,2,3))

    return loss_conf


def calculate_loss_xywh_yolov3_v2(pred_box_xy, pred_box_wh, true_box_xy,
                                  true_box_wh, anchor_box, net_factor, object_mask, batch_size_f):
    # per experiencor code
    box_loss_scale = tf.exp(true_box_wh) * anchor_box / net_factor
    box_loss_scale = tf.expand_dims(2 - box_loss_scale[..., 0] * box_loss_scale[..., 1], axis=4)

    # per github.com/david8862
    xy_delta = tf.expand_dims(binary_crossentropy(true_box_xy, pred_box_xy, from_logits=True), -1)
    xy_loss = object_mask * box_loss_scale * xy_delta

    wh_delta = tf.expand_dims(tf.square(true_box_wh-pred_box_wh),-1)
    wh_loss = object_mask * box_loss_scale * 0.5

    xy_loss = tf.reduce_sum(xy_loss) / batch_size_f
    wh_loss = tf.reduce_sum(wh_loss) / batch_size_f

    location_loss = xy_loss + wh_loss
    return location_loss



def get_intersect_area(pred_box_xy, pred_box_wh, anchors, grid_factor, net_factor, true_boxes=None):
    '''
    == INPUT ==
    true_xy,pred_xy, true_wh and pred_wh must have the same shape length

    p1 : pred_mins = (px1,py1)
    p2 : pred_maxs = (px2,py2)
    t1 : true_mins = (tx1,ty1)
    t2 : true_maxs = (tx2,ty2)
                 p1______________________
                 |      t1___________   |
                 |       |           |  |
                 |_______|___________|__|p2
                         |           |rmax
                         |___________|
                                      t2
    intersect_mins : rmin = t1  = (tx1,ty1)
    intersect_maxs : rmax = (rmaxx,rmaxy)
    intersect_wh   : (rmaxx - tx1, rmaxy - ty1)

    '''
    # then, ignore the boxes which have good overlap with some true box
    true_xy = true_boxes[..., 0:2] / grid_factor
    true_wh = true_boxes[..., 2:4] / net_factor

    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)
    pred_wh = tf.expand_dims(tf.exp(pred_box_wh) * anchors / net_factor, 4)

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)
    return (iou_scores)

def debug_online_statistics(true_box_xy, true_box_wh, true_box_class, true_box_conf,
                            pred_box_xy, pred_box_wh, pred_box_class, pred_box_conf,
                            anchors, grid_factor, net_factor, object_mask):
    true_xy = true_box_xy / grid_factor
    true_wh = tf.exp(true_box_wh) * anchors / net_factor

    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    pred_xy = pred_box_xy / grid_factor
    pred_wh = tf.exp(pred_box_wh) * anchors / net_factor

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)
    iou_scores = object_mask * tf.expand_dims(iou_scores, 4)

    count = tf.reduce_sum(object_mask)
    count_noobj = tf.reduce_sum(1 - object_mask)
    detect_mask = tf.cast((pred_box_conf * object_mask) >= 0.5, tf.float32)
    class_mask = tf.expand_dims(tf.cast(tf.equal(tf.argmax(pred_box_class, -1), true_box_class), tf.float32), 4)
    recall50 = tf.reduce_sum(tf.cast(iou_scores >= 0.5, tf.float32) * detect_mask * class_mask) / (count + 1e-3)
    recall75 = tf.reduce_sum(tf.cast(iou_scores >= 0.75, tf.float32) * detect_mask * class_mask) / (count + 1e-3)
    avg_iou = tf.reduce_sum(iou_scores) / (count + 1e-3)
    avg_obj = tf.reduce_sum(pred_box_conf * object_mask) / (count + 1e-3)
    avg_noobj = tf.reduce_sum(pred_box_conf * (1 - object_mask)) / (count_noobj + 1e-3)
    avg_cat = tf.reduce_sum(object_mask * class_mask) / (count + 1e-3)

    return (recall50, recall75, avg_iou, avg_obj, avg_noobj, avg_cat, count)

"""
Experiencor loss function as is
https://github.com/experiencor/keras-yolo3/blob/768c524f277adbfd26c2f44d73cb1826bbaf2d10/yolo.py
"""
class YoloLayer(Layer):
    def __init__(self, anchors, max_grid, batch_size, warmup_batches, ignore_thresh,
                 grid_scale, obj_scale, noobj_scale, xywh_scale, class_scale,
                 **kwargs):
        # make the model settings persistent
        self.ignore_thresh = ignore_thresh
        self.warmup_batches = warmup_batches
        self.anchors = tf.constant(anchors, dtype='float', shape=[1, 1, 1, 3, 2])
        self.grid_scale = grid_scale
        self.obj_scale = obj_scale
        self.noobj_scale = noobj_scale
        self.xywh_scale = xywh_scale
        self.class_scale = class_scale
        self.debug = False

        # make a persistent mesh grid
        max_grid_h, max_grid_w = max_grid

        cell_x = tf.cast(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)), tf.float32)
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))
        self.cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [batch_size, 1, 1, 3, 1])

        super(YoloLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(YoloLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        input_image, y_pred, y_true, true_boxes = x

        # adjust the shape of the y_predict [batch, grid_h, grid_w, 3, 4+1+nb_class]
        y_pred = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))

        # initialize the masks
        object_mask = tf.expand_dims(y_true[..., 4], 4)

        # the variable to keep track of number of batches processed
        batch_seen = tf.Variable(0.)

        # compute grid factor and net factor
        grid_h = tf.shape(y_true)[1]
        grid_w = tf.shape(y_true)[2]
        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1, 1, 1, 1, 2])

        net_h = tf.shape(input_image)[1]
        net_w = tf.shape(input_image)[2]
        net_factor = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1, 1, 1, 1, 2])

        """
        Adjust prediction
        """
        pred_box_xy = (self.cell_grid[:, :grid_h, :grid_w, :, :] + tf.sigmoid(y_pred[..., :2]))  # sigma(t_xy) + c_xy
        pred_box_wh = y_pred[..., 2:4]  # t_wh
        pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[..., 4]), 4)  # adjust confidence
        pred_box_class = y_pred[..., 5:]  # adjust class probabilities

        """
        Adjust ground truth
        """
        true_box_xy = y_true[..., 0:2]  # (sigma(t_xy) + c_xy)
        true_box_wh = y_true[..., 2:4]  # t_wh
        true_box_conf = tf.expand_dims(y_true[..., 4], 4)
        true_box_class = tf.argmax(y_true[..., 5:], -1)

        """
        Compare each predicted box to all true boxes
        """
        # initially, drag all objectness of all boxes to 0
        conf_delta = pred_box_conf - 0

        batch_size_tf = tf.shape(pred_box_xy)[0]
        batch_size_f = tf.cast(batch_size_tf, tf.float32)

        # then, ignore the boxes which have good overlap with some true box
        true_xy = true_boxes[..., 0:2] / grid_factor
        true_wh = true_boxes[..., 2:4] / net_factor

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)
        pred_wh = tf.expand_dims(tf.exp(pred_box_wh) * self.anchors / net_factor, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)

        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_delta *= tf.expand_dims(tf.cast(best_ious < self.ignore_thresh, tf.float32), 4)

        """
        Warm-up training
        """
        batch_seen = tf.compat.v1.assign_add(batch_seen, 1.)

        true_box_xy, true_box_wh, xywh_mask = tf.cond(tf.less(batch_seen, self.warmup_batches + 1),
                                                      lambda: [true_box_xy + (
                                                                  0.5 + self.cell_grid[:, :grid_h, :grid_w, :, :]) * (
                                                                           1 - object_mask),
                                                               true_box_wh + tf.zeros_like(true_box_wh) * (
                                                                           1 - object_mask),
                                                               tf.ones_like(object_mask)],
                                                      lambda: [true_box_xy,
                                                               true_box_wh,
                                                               object_mask])

        """
        Compare each true box to all anchor boxes
        """

        loss_location = calculate_loss_xywh_yolov3(pred_box_xy, pred_box_wh, true_box_xy,
                                   true_box_wh, true_box_conf, self.xywh_scale, self.anchors,
                                    net_factor, xywh_mask, object_mask, batch_size_f)
        # loss_location = calculate_loss_xywh_yolov3_v2(pred_box_xy, pred_box_wh, true_box_xy, true_box_xy,
        #                                               self.anchors, net_factor, object_mask, batch_size_f)   results into huge loca_loss like total_loss_loc: 2908996303205477618506792691564544.0000
        loss_conf = calc_loss_conf_yolov3(object_mask, true_box_conf, pred_box_conf, conf_delta, self.obj_scale,
                                          self.noobj_scale, batch_size_f)
        loss_class = calculate_loss_class_yolov3(object_mask, true_box_class, pred_box_class,
                                                 self.class_scale, batch_size_f)

        loss = loss_location + loss_conf + loss_class
        #loss_location = loss_xy + loss_wh

        # if self.debug:
        #     """
        #     Compute some online statistics
        #     """
        #     recall50, recall75, avg_iou, avg_obj, avg_noobj, avg_cat, count = debug_online_statistics(true_box_xy,
        #                                                                                               true_box_wh,
        #                                                                                               true_box_class,
        #                                                                                               true_box_conf,
        #                                                                                               pred_box_xy,
        #                                                                                               pred_box_wh,
        #                                                                                               pred_box_class,
        #                                                                                               pred_box_conf,
        #                                                                                               self.anchors,
        #                                                                                               grid_factor,
        #                                                                                               net_factor,
        #                                                                                               object_mask)
        #     loss = tf.Print(loss, [grid_h, avg_obj], message='avg_obj \t\t', summarize=1000)
        #     loss = tf.Print(loss, [grid_h, avg_noobj], message='avg_noobj \t\t', summarize=1000)
        #     loss = tf.Print(loss, [grid_h, avg_iou], message='avg_iou \t\t', summarize=1000)
        #     loss = tf.Print(loss, [grid_h, avg_cat], message='avg_cat \t\t', summarize=1000)
        #     loss = tf.Print(loss, [grid_h, recall50], message='recall50 \t', summarize=1000)
        #     loss = tf.Print(loss, [grid_h, recall75], message='recall75 \t', summarize=1000)
        #     loss = tf.Print(loss, [grid_h, count], message='count \t', summarize=1000)
        #     loss = tf.Print(loss, [grid_h, tf.reduce_sum(loss_xy),
        #                            tf.reduce_sum(loss_wh),
        #                            tf.reduce_sum(loss_conf),
        #                            tf.reduce_sum(loss_class)], message='loss xy, wh, conf, class: \t', summarize=1000)

        return loss * self.grid_scale, loss_location, loss_conf, loss_class











##################






