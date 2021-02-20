import sys
from collections import OrderedDict

import numpy as np
import os
from PIL import Image
from MyMScProj.jmod.keypoint.centernet.generators.common import Generator
from MyMScProj.jmod.keypoint.centernet.utils.image import read_image_bgr


class BatchGenerator(Generator):
    def __init__(self, instances, base_dir, shuffle=True, **kwargs):

        self.instances = instances
        self.shuffle = shuffle
        self.base_dir = base_dir
        self.batch_size = kwargs.get('batch_size')

        self.classes = kwargs.get('labels')
        # print("classes", self.classes)

        self.image_names = []
        self.image_data = OrderedDict()

        #self.base_dir = base_dir
        for inst in self.instances:
            img_file = inst['filename']
            if img_file not in self.image_data:
                self.image_data[img_file] = []

            all_objs = inst['object']
            for obj in all_objs:
                x1, y1, x2, y2, class_name = obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], obj['name']
                # Check that the bounding box is valid.
                if x2 <= x1:
                    #raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(img_file,x2, x1))
                    continue
                if y2 <= y1:
                    #raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(img_file,y2, y1))
                    continue
                if class_name not in self.classes:
                    #raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(img_file, class_name, classes))
                    continue

                self.image_data[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})

        self.image_names = list(self.image_data.keys())
        # print("image_names", self.image_names)
        # print("image data:", self.image_data)
        #sys.exit()
        super(BatchGenerator, self).__init__(**kwargs)

    def __len__(self):
        return int(np.ceil(float(len(self.instances) / self.batch_size)))

    def size(self):
        return len(self.image_names)

    def num_classes(self):
        return len(self.labels)

    def has_label(self, label):
        """
        Return True if label is a known label.
        """
        return label in self.labels

    def image_path(self, image_index):
        """
        Returns the image path for image_index.
        """
        return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        """
        Compute the aspect ratio for an image with image_index.
        """
        # PIL is fast for metadata
        image = Image.open(self.image_path(image_index))
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        return read_image_bgr(self.image_path(image_index))

    def name_to_label(self, name):
        """
        Map name to label.
        """
        #print("name_to_label() name",name)
        return self.classes.index(name)
        #return self.classes[name]


    def load_annotations(self, image_index):
        """
        Load annotations for an image_index.
        """
        path = self.image_names[image_index]
        annotations = {'labels': np.empty((0,), dtype=np.int32), 'bboxes': np.empty((0, 4))}

        for idx, annot in enumerate(self.image_data[path]):
            annotations['labels'] = np.concatenate((annotations['labels'], [self.name_to_label(annot['class'])]))
            annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[
                float(annot['x1']),
                float(annot['y1']),
                float(annot['x2']),
                float(annot['y2']),
            ]]))
        #print("load_annotations {} for index {}".format( annotations, image_index))
        return annotations


