import seaborn as sb
import numpy as np
import cv2
import os
import csv
import sys
import pandas as pd
from matplotlib import pyplot as plt
from random import sample


class Preprocess(object):
    def __init__(self,annot_csv_file, img_dir, labels=[]):
        self.annot_csv_file = annot_csv_file
        self.img_dir = img_dir
        self.labels = labels

    def prepare_annoted_dict(self):
        '''
        output:
        - Each element of the train_image is a dictionary containing the annoation infomation of an image.
        - seen_data_labels is the dictionary containing
                (key, value) = (the object class, the number of objects found in the images)
        '''
        all_images=[]
        seen_labels={}
        data_label={}
        with open(self.annot_csv_file,'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                # format of each row in annot_csv_file file is like
                # 00000000,pickup_truck,213,34,255,50
                img_name = row[0]
                bbox = {}
                bbox['class'] = row[1]

                #if len(row) == 6:
                bbox['bbox'] = np.array(row[2:]).astype('int32')
                if img_name in data_label:
                    data_label[img_name].append(bbox)
                else:
                    data_label[img_name] = [bbox]

        f.close()

        # iterate images from the img_dir and populate the all_images data structure
        for img_name in os.listdir(self.img_dir):
            #if img_name != "00000004.jpg" :
            #    continue
            img_name = img_name.split(".",1)[0]
            img_attr = {'object':[]}

            cannonical_path = os.path.join(self.img_dir, img_name + '.jpg')

            if os.path.exists(cannonical_path):
                img_attr['filename'] = cannonical_path
                (height, width, channel) = cv2.imread(cannonical_path).shape
                #print("height, width, channel", height, width, channel)
                img_attr['height'] = height
                img_attr['width'] = width

                bboxes = data_label[img_name]
                #print(bboxes)
                for bbox in bboxes:
                    obj={}
                    obj['name'] = bbox['class']
                    pts = bbox['bbox']
                    obj['xmin'] = int(pts[0])
                    obj['ymin'] = int(pts[1])
                    obj['xmax'] = int(pts[2]) # (x + width)
                    obj['ymax'] = int(pts[3]) # (y + height)
                    if obj['xmax'] < obj['xmin'] or obj['ymax'] < obj['ymin']:
                        continue
                    img_attr['object'] += [obj]

                    if obj['name'] in seen_labels:
                        seen_labels[obj['name']] +=1
                    else:
                        seen_labels[obj['name']] = 1

                    #print("object: ", obj)

            if len(img_attr['object']) > 0:
                all_images += [img_attr]
        ###
        return all_images, seen_labels

    def plt_train_dist(self, seen_train_labels, train_ds_cnt):
        y_pos = np.arange(len(seen_train_labels))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.barh(y_pos, list(seen_train_labels.values()))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(seen_train_labels.keys()))
        ax.set_title("The total number of objects = {} in {} images".format(
                np.sum(list(seen_train_labels.values())), train_ds_cnt))
        plt.show()

    def standardize_image_hw_with_bbox_hw(self, annot_ds):
        relative_wh = []
        for item in annot_ds:
            img_h = float(item['height'])
            img_w = float(item['width'])

            for obj in item['object']:
                rel_w = (obj['xmax'] - obj['xmin']) / img_w  # make the width range between [0,GRID_W)
                rel_h = (obj['ymax'] - obj['ymin']) / img_h  # make the width range between [0,GRID_H)
                temp = [rel_w, rel_h]
                relative_wh.append(temp)

        relative_wh = np.array(relative_wh)
        print("clustering feature data is ready. shape = (N object, width and height) =  {}".format(
                relative_wh.shape))
        return relative_wh

    def create_test_instances(self, annon_insts=None, seen_labels=None, sampling=None):
        test_insts = annon_insts
        test_labels = seen_labels
        if not len(test_insts) > 0:
            test_insts, test_labels = self.prepare_annoted_dict()

        if sampling is not None:
            test_insts = sample(test_insts, sampling)

        np.random.seed(0)
        np.random.shuffle(test_insts)

        return test_insts, test_labels


    def create_training_instances(self, annon_insts, seen_labels, sampling=None):
        train_insts = annon_insts
        train_labels = seen_labels

        if not len(train_insts) > 0:
            train_insts, train_labels = self.prepare_annoted_dict()

        # check if sampling is there
        if sampling is not None:
            train_insts = sample(train_insts, sampling)

        # split train_insts into train and validation sets
        train_valid_splits = int(0.8*len(train_insts))
        np.random.seed(0)
        np.random.shuffle(train_insts)
        np.random.seed()

        valid_insts = train_insts[train_valid_splits:]
        train_insts = train_insts[:train_valid_splits]

        # compare the seen labels with the given labels in config.json
        if len(train_labels) > 0:
            overlap_labels = set(train_labels).intersection(set(train_labels.keys()))

            print('Seen labels: \t' + str(train_labels) + '\n')
            print('Given labels: \t' + str(self.labels))

            # return None, None, None if some given label is not in the dataset
            if len(overlap_labels) < len(self.labels):
                print('Some labels have no annotations! Please revise the list of labels in the config.json.')
                return None, None, None
        else:
            print('No labels are provided. Train on all seen labels.')
            print(train_labels)
            self.labels = train_labels.keys()

        max_box_per_image = max([len(inst['object']) for inst in (valid_insts + train_insts)])
                          
        return train_insts, valid_insts, sorted(self.labels), max_box_per_image







####
