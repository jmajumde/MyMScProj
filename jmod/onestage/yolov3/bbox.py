import numpy as np
from matplotlib import pyplot as plt


class kmeans(object):
    def __init__(self,bbox):
        self.boxes = bbox
        self.dist = np.median
        self.seed = 1

    def iou(self, box, clusters):
        '''
        :param box:      np.array of shape (2,) containing w and h
        :param clusters: np.array of shape (N cluster, 2)
        '''
        x = np.minimum(clusters[:, 0], box[0])
        y = np.minimum(clusters[:, 1], box[1])

        intersection = x * y
        box_area = box[0] * box[1]
        cluster_area = clusters[:, 0] * clusters[:, 1]

        iou_ = intersection / (box_area + cluster_area - intersection)

        return iou_

    def run_kmeans(self,k):
        """
        Calculates k-means clustering with the Intersection over Union (IoU) metric.
        :param boxes: numpy array of shape (r, 2), where r is the number of rows
        :param k: number of clusters
        :param dist: distance function
        :return: numpy array of shape (k, 2)
        """
        rows = self.boxes.shape[0]

        distances     = np.empty((rows, k)) ## N row x N cluster
        last_clusters = np.zeros((rows,))

        np.random.seed(self.seed)

        # initialize the cluster centers to be k items
        clusters = self.boxes[np.random.choice(rows, k, replace=False)]

        while True:
            # Step 1: allocate each item to the closest cluster centers
            for icluster in range(k):
                # I made change to lars76's code here to make the code faster
                distances[:,icluster] = 1 - self.iou(self.boxes, clusters[icluster])

            nearest_clusters = np.argmin(distances, axis=1)

            if (last_clusters == nearest_clusters).all():
                break

            # Step 2: calculate the cluster centers as mean (or median) of all the cases in the clusters.
            for cluster in range(k):
                clusters[cluster] = self.dist(self.boxes[nearest_clusters == cluster], axis=0)

            last_clusters = nearest_clusters

        return clusters,nearest_clusters,distances


    def plot_elbow_curve(self,res, kmax):
        plt.figure(figsize=(8,8))
        plt.plot(np.arange(2,kmax),
            [1 - res[k]["WithinClusterMeanDist"] for k in range(2,kmax)],"o-")
        plt.title("within cluster mean of {}".format(self.dist))
        plt.ylabel("mean IOU")
        plt.xlabel("N clusters (= N anchor boxes)")
        plt.show()


class BoundingBox(object):
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=[]):
        self.xmin, self.ymin = xmin, ymin
        self.xmax, self.ymax = xmax, ymax

        # objectness or confidence score is used during inference probability
        # to indicate object is detected or not based on the probability score
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.score



class AnchorBox(object):
    def __init__(self, probable_anchor_array=None):
        self.anchors=None
        if probable_anchor_array is not None:
            self.set_anchors_bbox(probable_anchor_array)

    def _interval_overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3

    def set_anchors_bbox(self, probable_anchor_array):
        self.anchors = [
            BoundingBox(0, 0, probable_anchor_array[2 * i], probable_anchor_array[2 * i + 1])
            for i in range(int(len(probable_anchor_array) // 2))]

    def get_anchors_bbox(self):
        return self.anchors

    def bbox_iou(self, box1, box2):
        intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

        intersect = intersect_w * intersect_h

        w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
        w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

        union = w1 * h1 + w2 * h2 - intersect

        return float(intersect) / union

    def find(self, center_w, center_h):
        # find the anchor that best predicts this box
        best_anchor = None
        max_index = -1
        maxiou = -1
        # each Anchor box is specialized to have a certain shape.
        # e.g., flat large rectangle, or small square
        shifted_box = BoundingBox(0, 0, center_w, center_h)

        ##  For given object, find the best anchor box!
        for i in range(len(self.anchors)):  ## run through each anchor box
            anchor = self.anchors[i]
            iou = self.bbox_iou(shifted_box, anchor)
            if maxiou < iou:
                best_anchor = anchor
                max_index = i
                maxiou = iou
        return (best_anchor, max_index, maxiou)

    """
    Util to do Non-Max Suppression
    """
    def do_nms(self, boxes, nms_thresh):
        if len(boxes) > 0:
            nb_class = len(boxes[0].classes)
        else:
            return
        for c in range(nb_class):
            sorted_indices = np.argsort([-box.classes[c] for box in boxes])
            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]
                if boxes[index_i].classes[c] == 0: continue
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    if self.bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                        boxes[index_j].classes[c] = 0

        return boxes

    def rescale_centerxy(self, obj, config):
        '''
            obj:     dictionary containing xmin, xmax, ymin, ymax
            config : dictionary containing IMAGE_W, GRID_W, IMAGE_H and GRID_H
            where IMAGE_W = 416, IMAGE_H = 416 and GRID_W = 13, GRID_H = 13 default

            determine the position of the bounding box on the grid
        '''
        center_x = 0.5 * (obj['xmin'] + obj['xmax'])
        #center_x = center_x / float(config['IMAGE_W']) * config['GRID_W']
        center_x = center_x / (float(config['IMAGE_W']) / config['GRID_W'])

        center_y = 0.5 * (obj['ymin'] + obj['ymax'])
        #center_y = center_y / float(config['IMAGE_H']) * config['GRID_H']
        center_y = center_y / (float(config['IMAGE_H']) / config['GRID_H'])
        return(round(center_x,3), round(center_y,3))  # this center_x, center_y is per grid scale as YOLO wants

    def rescale_centerwh(self, obj, config, max_anchor=None):
        '''
        obj:     dictionary containing xmin, xmax, ymin, ymax
        config : dictionary containing IMAGE_W, GRID_W, IMAGE_H and GRID_H
        where IMAGE_W = 416, IMAGE_H = 416 and GRID_W = 13, GRID_H = 13 default

        determine the sizes of the bounding box
        '''
        if max_anchor is None:
            max_index = -1
            max_iou = -1
            shifted_box=BoundingBox(0,0,(obj['xmax']-obj['xmin']),(obj['ymax']-obj['ymin']))
            for i in range(len(self.anchors)):
                anchor = self.anchors[i]
                # print("anchor max={}, min={}".format(anchor.xmax, anchor.xmin))
                iou = round(self.bbox_iou(shifted_box, anchor), 3)

                if max_iou < iou:
                    max_anchor = anchor
                    max_index = i
                    max_iou = iou

        #center_w = np.log(obj['xmax'] - obj['xmin']) / float(max_anchor.xmax)
        #center_h = np.log(obj['ymax'] - obj['ymin']) / float(max_anchor.ymax)
        center_w = (obj['xmax'] - obj['xmin']) / (float(config['IMAGE_W']) / config['GRID_W'])
        center_h = (obj['ymax'] - obj['ymin']) / (float(config['IMAGE_H']) / config['GRID_H'])
        return (round(center_w,3), round(center_h, 3))









####
