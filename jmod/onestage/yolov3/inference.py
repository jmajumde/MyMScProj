import sys
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
base_path = '/home/jmajumde/PGDDS-IIITB/MyPractice'
proj_dir_path = os.path.join(base_path,"MyMScProj")
dataset_base_path = '/mywork/PGDDS-IIITB/MyDatasets'

sys.path.append(base_path)
sys.path.append(proj_dir_path)
import operator
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from MyMScProj.jmod.onestage.yolov3.bbox import BoundingBox, AnchorBox
from MyMScProj.jmod.utils import sigmoid, softmax
from collections import OrderedDict

def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w)/new_w) < (float(net_h)/new_h):
        new_h = (new_h * net_w)//new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h)//new_h
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image[:,:,::-1]/255., (new_w, new_h))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[(net_h-new_h)//2:(net_h+new_h)//2, (net_w-new_w)//2:(net_w+new_w)//2, :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image

"""
Decode yolo network output
The output will be in form of 3 scales detection 13x13, 26x26 and 52x52
The predicted output has encoded output vector of form (S x S x no-of-anchor x (bbox-vector + objecetness score + classes)). 
So besically  number is all inclusive of 'no-of-anchor x (bbox-vector + objecetness score + classes)' which 
eventually be '3 x ( 4 + 1 + 11 )'
"""
def decode_yolo_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    #grid_h, grid_w, nb_box = netout.shape[:3]
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5

    boxes = []

    netout[..., :2] = sigmoid(netout[..., :2])
    #netout[..., 2:4] = sigmoid(netout[..., 2:4])
    netout[..., 4] = sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h * grid_w):
        row = i // grid_w
        col = i % grid_w

        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]

            if (objectness.all() <= obj_thresh): continue

            # last elements are class probabilities
            classes = netout[int(row)][int(col)][b][5:]
            #print(classes)
            if np.sum(classes) > 0:

            #print(objectness, row, col, b)
            # first 4 elements are x, y, w, and h
                x, y, w, h = netout[int(row)][int(col)][b][:4]
            #print(x, y, w, h)

                x = (col + x) / grid_w  # center position, unit: image width
            #print("x with grid scale: ",x)
                y = (row + y) / grid_h  # center position, unit: image height
            #print("y with grid scale: ",y)
            #w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
                w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            #print("w width: ", w)
            #h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height
                h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height
            #print("h height: ", h)

                box = BoundingBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
            #print(x, y, w, h)
            #box = BoundingBox(x , y , w, h, objectness, classes)
                if box.get_score() > obj_thresh:
                    boxes.append(box)

    return boxes



# bounding boxes will be stretched back into the shape of the original image
#will allow plotting the original image and draw the bounding boxes, hopefully detecting real objects.
# correct the sizes of the bounding boxes for the shape of the image
#correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w) / image_w) < (float(net_h) / image_h):
        new_w = net_w
        new_h = (image_h * net_w) / image_w
    else:
        new_h = net_w
        new_w = (image_w * net_h) / image_h

    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h

        # boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        # boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        # boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        # boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
        boxes[i].xmin = int(abs(boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int(abs(boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int(abs(boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int(abs(boxes[i].ymax - y_offset) / y_scale * image_h)

    return boxes


# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i] * 100)
            # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores


# draw all results
def draw_boxes(filename, v_boxes, v_labels, v_scores):
    # load the image
    data = plt.imread(filename)
    # plot the image
    plt.imshow(data)
    # get the context for drawing boxes
    ax = plt.gca()
    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='white')
        # draw the box
        ax.add_patch(rect)
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        plt.text(x1, y1, label, color='white')
    # show the plot
    plt.show()


def get_pred_classes_records(model,
             generator,
             obj_thresh=0.4,
             nms_thresh=0.45,
             net_h=416,
             net_w=416,):
    '''
        Do the predict with YOLO model on annotation images to get predict class dict
        predict class dict would contain image_name, coordinary and score, and
        sorted by score:
        pred_classes_records = {
            'car': [
                    ['000001.jpg','94,115,203,232',0.98],
                    ['000002.jpg','82,64,154,128',0.93],
                    ...
                   ],
            ...
        }
        '''
    pred_classes_records = OrderedDict()
    for i in range(generator.size()):
        raw_image = [generator.load_image(i)]
        image_file_path =  generator.get_filename(img_indx=i)
        #print("processing image file", image_file_path)
        # make the boxes and the labels
        pred_boxes = get_yolo_boxes(model, raw_image, net_h, net_w, generator.get_anchors(), obj_thresh, nms_thresh)[0]
        image_name = os.path.basename(image_file_path)
        pred_scores = np.array([box.get_score() for box in pred_boxes])
        pred_labels = np.array([box.label for box in pred_boxes])

        for box in pred_boxes:
            label_idx = box.label
            pred_class_name = generator.get_class_names()[label_idx]
            xmin, ymin, xmax, ymax = box.xmin, box.ymin, box.xmax, box.ymax
            coordinate = "{},{},{},{}".format(xmin, ymin, xmax, ymax)
            pred_score = box.get_score()

            # append or add predict class item
            record = [os.path.basename(image_name), coordinate, pred_score]
            if pred_class_name in pred_classes_records:
                pred_classes_records[pred_class_name].append(record)
            else:
                pred_classes_records[pred_class_name] = list([record])

    # sort pred_classes_records for each class according to score
    for pred_class_list in pred_classes_records.values():
        pred_class_list.sort(key=lambda ele: ele[2], reverse=True)

    return pred_classes_records


def annotation_parse(generator):
    '''
        parse annotation lines to get image dict and ground truth class dict
        image dict would be like:
        annotation_records = {
            '/path/to/000001.jpg': {'100,120,200,235':'dog', '85,63,156,128':'car', ...},
            ...
        }
        ground truth class dict would be like:
        classes_records = {
            'car': [
                    ['000001.jpg','100,120,200,235'],
                    ['000002.jpg','85,63,156,128'],
                    ...
                   ],
            ...
        }
        '''
    class_names = generator.get_class_names()
    annotations = generator.get_instances()
    annotation_records = OrderedDict()
    classes_records = OrderedDict({class_name: [] for class_name in class_names})
    #print(classes_records)
    for annot in annotations:
        tmp_annon_vals = {}
        image_name = os.path.basename(annot['filename'])
        for obj in annot['object']:
            xmin, ymin, xmax, ymax = obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']
            coordinate = "{},{},{},{}".format(xmin, ymin, xmax, ymax)
            class_name = obj['name']
            record = [os.path.basename(image_name), coordinate]

            # populate classes_records
            if class_name in classes_records:
                classes_records[class_name].append(record)
            else:
                classes_records[class_name] = list([record])

            # populate annot records
            tmp_annon_vals[coordinate] = class_name
        annotation_records[annot['filename']] = tmp_annon_vals

    return annotation_records, classes_records


def compute_mAP_PascalVOC(annotation_records, gt_classes_records, pred_classes_records, class_names, iou_threshold, show_result=True):
    '''
    Compute PascalVOC style mAP
    '''
    APs = {}
    count_true_positives = {class_name: 0 for class_name in list(gt_classes_records.keys())}
    #get AP value for each of the ground truth classes
    for _, class_name in enumerate(class_names):
        #if there's no gt obj for a class, record 0
        if class_name not in gt_classes_records:
            APs[class_name] = 0.
            continue
        gt_records = gt_classes_records[class_name]
        #if we didn't detect any obj for a class, record 0
        if class_name not in pred_classes_records:
            APs[class_name] = 0.
            continue
        pred_records = pred_classes_records[class_name]

        # print("gt_records", gt_records)
        # print("pred_records", pred_records)

        ap, true_positive_count = calc_AP(gt_records, pred_records, class_name, iou_threshold, show_result)
        APs[class_name] = ap
        count_true_positives[class_name] = true_positive_count

    #sort AP result by value, in descending order
    APs = OrderedDict(sorted(APs.items(), key=operator.itemgetter(1), reverse=True))

    #get mAP percentage value
    #mAP = np.mean(list(APs.values()))*100
    mAP = get_mean_metric(APs, gt_classes_records)

    #get GroundTruth count per class
    gt_counter_per_class = {}
    for (class_name, info_list) in gt_classes_records.items():
        gt_counter_per_class[class_name] = len(info_list)

    #get Precision count per class
    pred_counter_per_class = {class_name: 0 for class_name in list(gt_classes_records.keys())}
    for (class_name, info_list) in pred_classes_records.items():
        pred_counter_per_class[class_name] = len(info_list)


    #get the precision & recall
    precision_dict = {}
    recall_dict = {}
    for (class_name, gt_count) in gt_counter_per_class.items():
        if (class_name not in pred_counter_per_class) or (class_name not in count_true_positives) or pred_counter_per_class[class_name] == 0:
            precision_dict[class_name] = 0.
        else:
            precision_dict[class_name] = float(count_true_positives[class_name]) / pred_counter_per_class[class_name]

        if class_name not in count_true_positives or gt_count == 0:
            recall_dict[class_name] = 0.
        else:
            recall_dict[class_name] = float(count_true_positives[class_name]) / gt_count

    #get mPrec, mRec
    #mPrec = np.mean(list(precision_dict.values()))*100
    #mRec = np.mean(list(recall_dict.values()))*100
    mPrec = get_mean_metric(precision_dict, gt_classes_records)
    mRec = get_mean_metric(recall_dict, gt_classes_records)


    if show_result:
        plot_Pascal_AP_result(len(annotation_records), count_true_positives, len(gt_classes_records),
                                  gt_counter_per_class, pred_counter_per_class,
                                  precision_dict, recall_dict, mPrec, mRec,
                                  APs, mAP, iou_threshold)
        #show result
        print('\nPascal VOC AP evaluation')
        for (class_name, AP) in APs.items():
            print('%s: AP %.4f, precision %.4f, recall %.4f' % (class_name, AP, precision_dict[class_name], recall_dict[class_name]))
        print('mAP@IoU=%.2f result: %f' % (iou_threshold, mAP))
        print('mPrec@IoU=%.2f result: %f' % (iou_threshold, mPrec))
        print('mRec@IoU=%.2f result: %f' % (iou_threshold, mRec))

    #return mAP percentage value
    return mAP, APs



def calc_AP(gt_records, pred_records, class_name, iou_threshold, show_result):
    '''
    Calculate AP value for one class records
    Param
         gt_records: ground truth records list for one class, with format:
                     [
                      ['image_file', 'xmin,ymin,xmax,ymax'],
                      ['image_file', 'xmin,ymin,xmax,ymax'],
                      ...
                     ]
         pred_records: predict records for one class, with format (in score descending order):
                     [
                      ['image_file', 'xmin,ymin,xmax,ymax', score],
                      ['image_file', 'xmin,ymin,xmax,ymax', score],
                      ...
                     ]
    Return
         AP value for the class
    '''

    #print("pred_records",pred_records)
    # append usage flag in gt_records for matching gt search
    gt_records = [gt_record + ['unused'] for gt_record in gt_records]
    #print("gt_records", gt_records)

    # prepare score list for generating P-R html page
    scores = [pred_record[2] for pred_record in pred_records]

    # init true_positive and false_positive list
    nd = len(pred_records)  # number of predict data
    true_positive = [0] * nd
    false_positive = [0] * nd
    true_positive_count = 0
    # assign predictions to ground truth objects
    for idx, pred_record in enumerate(pred_records):

        # filter out gt record from same image
        image_gt_records = [ gt_record for gt_record in gt_records if gt_record[0] == pred_record[0]]
        #print("image_gt_records",image_gt_records)

        i = match_gt_box(pred_record, image_gt_records, iou_threshold=iou_threshold)
        #i = match_gt_box(pred_record, image_gt_records, iou_threshold=iou_threshold)
        if i != -1:
            # find a valid gt obj to assign, set
            # true_positive list and mark image_gt_records.
            #
            # trick: gt_records will also be marked
            # as 'used', since image_gt_records is a
            # reference list
            image_gt_records[i][2] = 'used'
            #gt_records[i][2] = 'used'
            true_positive[idx] = 1
            true_positive_count += 1
        else:
            false_positive[idx] = 1

    # compute precision/recall
    rec, prec = get_rec_prec(true_positive, false_positive, gt_records)
    ap, mrec, mprec = voc_ap(rec, prec)
    if show_result:
        draw_rec_prec(rec, prec, mrec, mprec, class_name, ap)
        #generate_rec_prec_html(mrec, mprec, scores, class_name, ap)

    return ap, true_positive_count


def plot_Pascal_AP_result(count_images, count_true_positives, num_classes,
                          gt_counter_per_class, pred_counter_per_class,
                          precision_dict, recall_dict, mPrec, mRec,
                          APs, mAP, iou_threshold):
    '''
     Plot the total number of occurences of each class in the ground-truth
    '''
    window_title = "Ground-Truth Info"
    plot_title = "Ground-Truth\n" + "(" + str(count_images) + " files and " + str(num_classes) + " classes)"
    x_label = "Number of objects per class"
    output_path = os.path.join('result','Ground-Truth_Info.png')
    draw_plot_func(gt_counter_per_class, num_classes, window_title, plot_title, x_label, output_path, to_show=False, plot_color='forestgreen', true_p_bar='')

    '''
     Plot the total number of occurences of each class in the "predicted" folder
    '''
    window_title = "Predicted Objects Info"
    # Plot title
    plot_title = "Predicted Objects\n" + "(" + str(count_images) + " files and "
    count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(pred_counter_per_class.values()))
    plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
    # end Plot title
    x_label = "Number of objects per class"
    output_path = os.path.join('result','Predicted_Objects_Info.png')
    draw_plot_func(pred_counter_per_class, len(pred_counter_per_class), window_title, plot_title, x_label, output_path, to_show=False, plot_color='forestgreen', true_p_bar=count_true_positives)

    '''
     Draw mAP plot (Show AP's of all classes in decreasing order)
    '''
    window_title = "mAP"
    plot_title = "mAP@IoU={0}: {1:.2f}%".format(iou_threshold, mAP)
    x_label = "Average Precision"
    output_path = os.path.join('result','mAP.png')
    draw_plot_func(APs, num_classes, window_title, plot_title, x_label, output_path, to_show=False, plot_color='royalblue', true_p_bar='')

    '''
     Draw Precision plot (Show Precision of all classes in decreasing order)
    '''
    window_title = "Precision"
    plot_title = "mPrec@IoU={0}: {1:.2f}%".format(iou_threshold, mPrec)
    x_label = "Precision rate"
    output_path = os.path.join('result','Precision.png')
    draw_plot_func(precision_dict, len(precision_dict), window_title, plot_title, x_label, output_path, to_show=False, plot_color='royalblue', true_p_bar='')

    '''
     Draw Recall plot (Show Recall of all classes in decreasing order)
    '''
    window_title = "Recall"
    plot_title = "mRec@IoU={0}: {1:.2f}%".format(iou_threshold, mRec)
    x_label = "Recall rate"
    output_path = os.path.join('result','Recall.png')
    draw_plot_func(recall_dict, len(recall_dict), window_title, plot_title, x_label, output_path, to_show=False, plot_color='royalblue', true_p_bar='')


def get_mean_metric(metric_records, gt_classes_records):
    '''
    Calculate mean metric, but only count classes which have ground truth object
    Param
        metric_records: metric dict like:
            metric_records = {
                'aeroplane': 0.79,
                'bicycle': 0.79,
                    ...
                'tvmonitor': 0.71,
            }
        gt_classes_records: ground truth class dict like:
            gt_classes_records = {
                'car': [
                    ['000001.jpg','100,120,200,235'],
                    ['000002.jpg','85,63,156,128'],
                    ...
                    ],
                ...
            }
    Return
         mean_metric: float value of mean metric
    '''
    mean_metric = 0.0
    count = 0
    for (class_name, metric) in metric_records.items():
        if (class_name in gt_classes_records) and (len(gt_classes_records[class_name]) != 0):
            mean_metric += metric
            count += 1
    mean_metric = (mean_metric/count)*100 if count != 0 else 0.0
    return mean_metric


def get_yolo_boxes(model, images, net_h, net_w, anchors, obj_thresh, nms_thresh):
    image_h, image_w, _ = images[0].shape
    nb_images = len(images)
    #print("nb_images",nb_images)
    batch_input = np.zeros((nb_images, net_h, net_w, 3))

    # preprocess the input
    for i in range(nb_images):
        batch_input[i] = preprocess_input(images[i], net_h, net_w)

        # run the prediction
    batch_output = model.predict_on_batch(batch_input)
    #print([a.shape for a in batch_output])
    #print(batch_output)
    batch_boxes = [None] * nb_images

    for i in range(0,nb_images):
        yolos = [batch_output[0][i], batch_output[1][i], batch_output[2][i]]
        boxes = []

        # decode the output of the network
        for j in range(len(yolos)):
            yolo_anchors = anchors[(2 - j) * 6:(3 - j) * 6]  # config['model']['anchors']
            #print("decoding output for ",yolos[j])
            #print("using anchors {}, object threshold {}".format(yolo_anchors,obj_thresh))
            boxes += decode_yolo_netout(yolos[j], yolo_anchors, obj_thresh, net_h, net_w)

        #print("len of boxes", len(boxes))
        # for i in range(len(boxes)):
        #     xmin, xmax, ymin, ymax = boxes[i].xmin, boxes[i].xmax, boxes[i].ymin, boxes[i].ymax
        #     print("xmin {}, xmax {}, ymin {}, ymax {}".format(xmin, xmax, ymin, ymax))

        # correct the sizes of the bounding boxes
        correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

        # suppress non-maximal boxes
        AnchorBox().do_nms(boxes, nms_thresh)

        batch_boxes[i] = boxes

    return batch_boxes


def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def match_gt_box(pred_record, gt_records, iou_threshold=0.5):
    '''
    Search gt_records list and try to find a matching box for the predict box
    Param
         pred_record: with format ['image_file', 'xmin,ymin,xmax,ymax', score]
         gt_records: record list with format
                     [
                      ['image_file', 'xmin,ymin,xmax,ymax', 'usage'],
                      ['image_file', 'xmin,ymin,xmax,ymax', 'usage'],
                      ...
                     ]
         iou_threshold:
         pred_record and gt_records should be from same annotation image file
    Return
         matching gt_record index. -1 when there's no matching gt
    '''
    max_iou = 0.0
    max_index = -1
    #get predict box coordinate
    pred_box = [float(x) for x in pred_record[1].split(',')]

    for i, gt_record in enumerate(gt_records):
        #get ground truth box coordinate
        gt_box = [float(x) for x in gt_record[1].split(',')]
        iou = box_iou(pred_box, gt_box)
        #print("iou", iou)

        # if the ground truth has been assigned to other
        # prediction, we couldn't reuse it
        if iou > max_iou and gt_record[2] == 'unused' and pred_record[0] == gt_record[0]:
            max_iou = iou
            max_index = i

    # drop the prediction if couldn't match iou threshold
    if max_iou < iou_threshold:
        max_index = -1

    return max_index

def box_iou(pred_box, gt_box):
    '''
    Calculate iou for predict box and ground truth box
    Param
         pred_box: predict box coordinate
                   (xmin,ymin,xmax,ymax) format
         gt_box: ground truth box coordinate
                 (xmin,ymin,xmax,ymax) format
    Return
         iou value
    '''
    # get intersection box
    inter_box = [max(pred_box[0], gt_box[0]), max(pred_box[1], gt_box[1]), min(pred_box[2], gt_box[2]), min(pred_box[3], gt_box[3])]
    inter_w = max(0.0, inter_box[2] - inter_box[0] + 1)
    inter_h = max(0.0, inter_box[3] - inter_box[1] + 1)

    # compute overlap (IoU) = area of intersection / area of union
    pred_area = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
    gt_area = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
    inter_area = inter_w * inter_h
    union_area = pred_area + gt_area - inter_area
    return 0 if union_area == 0 else float(inter_area) / float(union_area)


def get_rec_prec(true_positive, false_positive, gt_records):
    '''
    Calculate precision/recall based on true_positive, false_positive
    result.
    '''
    cumsum = 0
    for idx, val in enumerate(false_positive):
        false_positive[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(true_positive):
        true_positive[idx] += cumsum
        cumsum += val

    rec = true_positive[:]
    for idx, val in enumerate(true_positive):
        rec[idx] = round((float(true_positive[idx]) / len(gt_records)),2) if len(gt_records) != 0 else 0

    prec = true_positive[:]
    for idx, val in enumerate(true_positive):
        prec[idx] = round(float(true_positive[idx]) / (false_positive[idx] + true_positive[idx]),2)

    return rec, prec

def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
        mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
      (goes from the end to the beginning)
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #   range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #   range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
    """
    # matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
      (numerical integration)
    """
    # matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre

def adjust_axes(r, t, fig, axes):
    """
     Plot - adjust axes
    """
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
    """
     Draw plot using Matplotlib
    """
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    #
    if true_p_bar != "":
        """
         Special case to draw in (green=true predictions) & (red=false predictions)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Predictions')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Predictions', left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            #   first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    else:
      plt.barh(range(n_classes), sorted_values, color=plot_color)
      """
       Write number on side of bar
      """
      fig = plt.gcf() # gcf - get current figure
      axes = plt.gca()
      r = fig.canvas.get_renderer()
      for i, val in enumerate(sorted_values):
          str_val = " " + str(val) # add a space before
          if val < 1.0:
              str_val = " {0:.2f}".format(val)
          t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
          # re-set axes to show number inside the figure
          if i == (len(sorted_values)-1): # largest bar
              adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15    # in percentage of the figure height
    bottom_margin = 0.05 # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()

#####


def draw_rec_prec(rec, prec, mrec, mprec, class_name, ap):
    """
     Draw plot
    """
    plt.plot(rec, prec, '-o')
    # add a new penultimate point to the list (mrec[-2], 0.0)
    # since the last line segment (and respective area) do not affect the AP value
    area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
    area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
    plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
    # set window title
    fig = plt.gcf() # gcf - get current figure
    fig.canvas.set_window_title('AP ' + class_name)
    # set plot title
    plt.title('class: ' + class_name + ' AP = {}%'.format(ap*100))
    #plt.suptitle('This is a somewhat long figure title', fontsize=16)
    # set axis titles
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # optional - set axes
    axes = plt.gca() # gca - get current axes
    axes.set_xlim([0.0,1.0])
    axes.set_ylim([0.0,1.05]) # .05 to give some extra space
    # Alternative option -> wait for button to be pressed
    #while not plt.waitforbuttonpress(): pass # wait for key display
    # Alternative option -> normal display
    #plt.show()
    # save the plot
    rec_prec_plot_path = os.path.join('result','classes')
    os.makedirs(rec_prec_plot_path, exist_ok=True)
    fig.savefig(os.path.join(rec_prec_plot_path, class_name + ".png"))
    plt.cla() # clear axes for next plot

import bokeh
import bokeh.io as bokeh_io
import bokeh.plotting as bokeh_plotting
def generate_rec_prec_html(mrec, mprec, scores, class_name, ap):
    """
     generate dynamic P-R curve HTML page for each class
    """
    # bypass invalid class
    if len(mrec) == 0 or len(mprec) == 0 or len(scores) == 0:
        return

    rec_prec_plot_path = os.path.join('result' ,'classes')
    os.makedirs(rec_prec_plot_path, exist_ok=True)
    bokeh_io.output_file(os.path.join(rec_prec_plot_path, class_name + '.html'), title='P-R curve for ' + class_name)

    # prepare curve data
    area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
    area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
    score_on_curve = [0.0] + scores[:-1] + [0.0] + [scores[-1]] + [1.0]
    source = bokeh.models.ColumnDataSource(data={
      'rec'      : area_under_curve_x,
      'prec' : area_under_curve_y,
      'score' : score_on_curve,
    })

    # prepare plot figure
    plt_title = 'class: ' + class_name + ' AP = {}%'.format(ap*100)
    plt = bokeh_plotting.figure(plot_height=200 ,plot_width=200, tools="", toolbar_location=None,
               title=plt_title, sizing_mode="scale_width")
    plt.background_fill_color = "#f5f5f5"
    plt.grid.grid_line_color = "white"
    plt.xaxis.axis_label = 'Recall'
    plt.yaxis.axis_label = 'Precision'
    plt.axis.axis_line_color = None

    # draw curve data
    plt.line(x='rec', y='prec', line_width=2, color='#ebbd5b', source=source)
    plt.add_tools(bokeh.models.HoverTool(
      tooltips=[
        ( 'score', '@score{0.0000 a}'),
        ( 'Prec', '@prec'),
        ( 'Recall', '@rec'),
      ],
      formatters={
        'rec'      : 'printf',
        'prec' : 'printf',
      },
      mode='vline'
    ))
    bokeh_io.save(plt)
    return







#####