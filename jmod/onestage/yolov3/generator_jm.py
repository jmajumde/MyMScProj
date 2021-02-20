from MyMScProj.jmod.onestage.yolov3.bbox import AnchorBox, BoundingBox
import numpy as np
from tensorflow.keras.utils import Sequence
from MyMScProj.jmod.onestage.yolov3.inputencoder import ImageEncoder
from MyMScProj.jmod.onestage.image import random_distort_image, random_flip, apply_random_scale_and_crop, correct_bounding_boxes
import cv2


#class BatchGenerator(object):
class BatchGenerator(Sequence):
    def __init__(self, instances, config, norm=None, shuffle=True):
        '''
        :param instances:
        :param config:
        :param norm:
        :param shuffle:
        '''
        self.config = config
        self.anchors = self.config['ANCHORS']
        self.labels = self.config['LABELS']
        self.noOfLabels = len(self.config['LABELS'])
        self.net_w = self.config['NET_W']
        self.net_h = self.config['NET_H']
        self.grid_w = self.config['GRID_W']
        self.grid_h = self.config['GRID_H']
        self.batch_size = self.config['BATCH_SIZE']
        self.max_box_per_image = self.config['TRUE_BOX_BUFFER']
        self.instances = instances
        self.norm = norm
        self.jitter = self.config['jitter']

        # 9 anchors with xmin/ymin as 0 and xmax/ymax as array elements in (pair)
        self.anchorBox = AnchorBox(self.anchors)

        self.shuffle = shuffle

        #print("I'm here...")
        if self.shuffle:
            np.random.shuffle(self.instances)

    def __len__(self):
        return int(np.ceil(float(len(self.instances) / self.batch_size)))

    def print_config(self):
        print(self.config)

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.instances)

    def __getitem__(self, idx):
        '''
                == input ==

                idx : non-negative integer value e.g., 0

                == output ==

                x_batch: The numpy array of shape  (BATCH_SIZE, IMAGE_H, IMAGE_W, N channels).

                    x_batch[iframe,:,:,:] contains a iframe-th frame of size  (IMAGE_H,IMAGE_W).

                y_batch:
                    The numpy array of shape  (BATCH_SIZE, GRID_H, GRID_W, BOX, 4 + 1 + N classes).
                    BOX = The number of anchor boxes.

                    y_batch[iframe,igrid_h,igrid_w,ianchor,:4] contains (center_x,center_y,center_w,center_h)
                    of ianchor-th anchor at  grid cell=(igrid_h,igrid_w) if the object exists in
                    this (grid cell, anchor) pair, else they simply contain 0.

                    y_batch[iframe,igrid_h,igrid_w,ianchor,4] contains 1 if the object exists in this
                    (grid cell, anchor) pair, else it contains 0.

                    y_batch[iframe,igrid_h,igrid_w,ianchor,5 + iclass] contains 1 if the iclass^th
                    class object exists in this (grid cell, anchor) pair, else it contains 0.


                b_batch:

                    The numpy array of shape (BATCH_SIZE, 1, 1, 1, TRUE_BOX_BUFFER, 4).

                    b_batch[iframe,1,1,1,ibuffer,ianchor,:] contains ibufferth object's
                    (center_x,center_y,center_w,center_h) in iframeth frame.

                    If ibuffer > N objects in iframeth frame, then the values are simply 0.

                    TRUE_BOX_BUFFER has to be some large number, so that the frame with the
                    biggest number of objects can also record all objects.

                    The order of the objects do not matter.

                    This is just a hack to easily calculate loss.

                '''

        # determine the first and the last indices of the batch
        l_bound = idx * self.batch_size
        r_bound = (idx + 1) * self.batch_size

        # in case user specified batch_size more than total number of train instances, handle that
        if r_bound > len(self.instances):
            r_bound = len(self.instances)
            l_bound = r_bound - self.batch_size

        ## prepare empty storage space: this will be output
        # input images
        x_batch = np.zeros((r_bound - l_bound, self.net_h, self.net_w, 3))

        # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        b_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.max_box_per_image, 4))

        # desired network output, in 3 different grid scale
        #y_batch = np.zeros((r_bound - l_bound, 1 * self.grid_h, 1 * self.grid_w, len(self.anchors) // 2, 4 + 1 + self.noOfLabels))
        y_batch13x13 = np.zeros((r_bound - l_bound, 1 * self.grid_h, 1 * self.grid_w, len(self.anchors) // 6, 4 + 1 + self.noOfLabels))
        y_batch26x26 = np.zeros((r_bound - l_bound, 2 * self.grid_h, 2 * self.grid_w, len(self.anchors) // 6, 4 + 1 + self.noOfLabels))
        y_batch52x52 = np.zeros((r_bound - l_bound, 4 * self.grid_h, 4 * self.grid_w, len(self.anchors) // 6, 4 + 1 + self.noOfLabels))
        multi_scale_yolos = [y_batch13x13, y_batch26x26, y_batch52x52]

        dummy_yolo_1 = np.zeros((r_bound - l_bound, 1))
        dummy_yolo_2 = np.zeros((r_bound - l_bound, 1))
        dummy_yolo_3 = np.zeros((r_bound - l_bound, 1))

        instance_cnt = 0;
        true_box_index = 0;

        for train_instance in self.instances[l_bound:r_bound]:
            imageReader = ImageEncoder(train_instance, self.net_h, self.net_w)
            image, all_objs = imageReader.fit()

            # augment input image and fix object's position and size
            #image, all_objs = self._aug_image(train_instance, self.net_h, self.net_w)

            print("all_objs:{}".format(all_objs))
            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.labels:
                    obj_index = (self.labels).index(obj['name'])
                    #print("obj_index: ", obj_index)
                    # # find the best anchor box for this object

                    shifted_w = obj['xmax'] - obj['xmin']
                    shifted_h = obj['ymax'] - obj['ymin']
                    #print("Determine the best anchor box for w: {}, h: {} ".format(shifted_w, shifted_h))
                    best_anchor, max_index, max_iou = self.anchorBox.find(shifted_w, shifted_h)
                    # max_iou = round(max_iou, 2)
                    #print("best anchor: {} with iou:{} ".format(max_index, max_iou))


                    # determine the yolo to be responsible for this bounding box
                    #print("max_index:{}".format(max_index))
                    y_batch = multi_scale_yolos[max_index // 3]
                    grid_h, grid_w = y_batch.shape[1:3]
                    #print("Modified grid_h={}, grid_w={}".format(grid_h, grid_w))

                    # determine the position of the bounding box on the grid
                    center_x = (obj['xmin'] + obj['xmax']) // 2
                    center_x = center_x / float(self.net_w) * grid_w  # sigma(t_x) + c_x
                    #center_x = center_x / (float(self.net_w) / grid_w)

                    center_y =  (obj['ymin'] + obj['ymax']) // 2
                    center_y = center_y / float(self.net_h) * grid_h  # sigma(t_y) + c_y
                    #center_y = center_y / (float(self.net_h) / grid_h)

                    # determine the sizes of the bounding box
                    #print("best_anchor.xmax {}, best_anchor.ymax {}".format(best_anchor.xmax,best_anchor.ymax))
                    w = np.log((obj['xmax'] - obj['xmin']) / float(best_anchor.xmax))  # t_w
                    h = np.log((obj['ymax'] - obj['ymin']) / float(best_anchor.ymax))  # t_h
                    #w = (obj['xmax'] - obj['xmin']) / float(self.net_w) * grid_w
                    #h = (obj['ymax'] - obj['ymin']) / float(self.net_h) * grid_h
                    #w = (obj['xmax'] - obj['xmin']) / float(self.net_w)
                    #h = (obj['ymax'] - obj['ymin']) / float(self.net_h)
                    #w = (obj['xmax'] - obj['xmin']) / (float(self.net_w) / grid_w)
                    #h = (obj['ymax'] - obj['ymin']) / (float(self.net_h) / grid_h)

                    box = [center_x, center_y, w, h]
                    #print("bounding box: {}".format(box))

                    # determine the location of the cell responsible for this object
                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))
                    #print("grid_x {}, grid_y {}".format(grid_x, grid_y))
                    #print("instance_cnt {}".format(instance_cnt))
                    # assign ground truth y, x, w, h, confidence and class probs to y_batch
                    if grid_x < grid_w and grid_y < grid_h:
                        y_batch[instance_cnt, grid_y, grid_x, max_index%3 ] = 0
                        y_batch[instance_cnt, grid_y, grid_x, max_index%3 , 0:4] = box  # center_x, center_y, w, h
                        y_batch[instance_cnt, grid_y, grid_x, max_index%3 , 4] = 1  # ground truth confidence
                        y_batch[instance_cnt, grid_y, grid_x, max_index%3 , 4 + 1 + obj_index] = 1  # class probability of object

                        # assign the true box to t_batch
                        true_box = [center_x, center_y, obj['xmax'] - obj['xmin'], obj['ymax'] - obj['ymin']]
                        b_batch[instance_cnt, 0, 0, 0, true_box_index] = true_box

                        true_box_index += 1
                        true_box_index = true_box_index % self.max_box_per_image

            # assign input image to x_batch
            x_batch[instance_cnt] = image # already normalized

            instance_cnt += 1

        return [x_batch, b_batch, y_batch13x13, y_batch26x26, y_batch52x52], [dummy_yolo_1, dummy_yolo_2, dummy_yolo_3]

    def _aug_image(self, instance, net_h, net_w):
        image_name = instance['filename']
        image = cv2.imread(image_name)  # RGB image

        if image is None: print('Cannot find ', image_name)
        image = image[:, :, ::-1]  # RGB image

        image_h, image_w, _ = image.shape

        # determine the amount of scaling and cropping
        dw = self.jitter * image_w
        dh = self.jitter * image_h
        #
        new_ar = (image_w + np.random.uniform(-dw, dw)) / (image_h + np.random.uniform(-dh, dh))
        scale = np.random.uniform(0.25, 2)
        #
        if (new_ar < 1):
            new_h = int(scale * net_h);
            new_w = int(net_h * new_ar);
        else:
            new_w = int(scale * net_w);
            new_h = int(net_w / new_ar);

        dx = int(np.random.uniform(0, net_w - new_w))
        dy = int(np.random.uniform(0, net_h - new_h))

        # apply scaling and cropping
        im_sized = apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy)
        # im_sized = cv2.resize(image, (net_w, net_h))
        # new_h, new_w, _ = im_sized.shape
        # dx = 0
        # dy = 0  # as didn't do any cropping, so no  delta

        # randomly distort hsv space
        im_sized = random_distort_image(im_sized)

        # randomly flip
        flip = np.random.randint(2)
        im_sized = random_flip(im_sized, flip)

        # correct the size and pos of bounding boxes
        all_objs = correct_bounding_boxes(instance['object'], new_w, new_h, net_w, net_h, dx, dy, image_w, image_h)

        return im_sized, all_objs


    def load_image(self, img_indx):
        return cv2.imread(self.instances[img_indx]['filename'])

    def get_annotations(self, img_indx):
        annots =[]
        for obj in self.instances[img_indx]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.labels.index(obj['name'])]
            annots += [annot]

        if len(annots) == 0:
            annots = [[]]

        return np.array(annots)

    def get_filename(self, img_indx):
        return self.instances[img_indx]['filename']

    def get_anchors(self):
        anchors=[]
        for anchor in self.anchorBox.get_anchors_bbox():
            anchors += [anchor.xmax, anchor.ymax]

        return anchors

    def num_classes(self):
        return len(self.labels)

    def get_class_names(self):
        return self.labels

    def get_instances(self):
        return self.instances

    def size(self):
        return len(self.instances)





