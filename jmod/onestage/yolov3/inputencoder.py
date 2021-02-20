import copy
import cv2
from PIL import Image
import numpy as np


class ImageEncoder(object):
    def __init__(self, train_inst, net_h, net_w):
        '''
        net_h : the height of the rescaled image, e.g., 416
        net_w : the width of the rescaled image, e.g., 416
        '''
        self.image_path = train_inst['filename']
        self.all_objs = train_inst['object']
        self.image = cv2.imread(self.image_path)
        #self.image = Image.open(self.image_path)
        #self.image_size = self.image.size
        self.target_shape = (net_w, net_h)
        #boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    def normalize(self, image):
        #print("image shape before normalize ", image.shape)
        return image / 255

    def encode_core(self):
        # resize the image to standard size
        image_resized = cv2.resize(self.image, self.target_shape )
        #if reorder_rgb:
        #    image = image[:, :, ::-1]
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        #print("resized image shape ", image.shape)
        return self.normalize(image_resized)
        return image_resized

    def fit(self):
        '''
        read in and resize the image, annotations are resized accordingly.

        -- Input --
        annot_ds_dict : dictionary containing filename, height, width and object

        {'object': [{'name': 'motorcycle',
    'xmin': 204,
    'ymin': 65,
    'xmax': 223,
    'ymax': 88},
   {'name': 'car', 'xmin': 105, 'ymin': 93, 'xmax': 183, 'ymax': 149},
   {'name': 'car', 'xmin': 629, 'ymin': 175, 'xmax': 682, 'ymax': 225}],
  'filename': '/mywork/PGDDS-IIITB/MyDatasets/MIO-TCD/MIO-TCD-Localization/train/00004115.jpg',
  'height': 480,
  'width': 720}

        '''
        # (image_h, image_w, c) = self.image.shape
        # if self.image is None: print('Cannot find ', self.image_path)

        new_image, padding_shape, offset = self.letterbox_resize(self.image, self.target_shape, return_padding_info=True)
        image_data = np.array(new_image)
        image_data = self.normalize(image_data)

        #reshape boxes
        # boxes = self.reshape_boxes(self.all_objs, src_shape=self.image_size, target_shape=self.model_input_size, padding_shape=padding_size,
        #                       offset=offset)


        # image_resized = self.encode_core()
        # new_h, new_w, _ = image_resized.shape
        #
        # # fix object's position and size
        # # correct sizes and positions
        copy_objs = copy.deepcopy(self.all_objs)
        target_w, target_h = self.target_shape
        #src_w, src_h = self.image_size
        src_w, src_h, _ =  self.image.shape
        padding_w, padding_h = padding_shape
        dx, dy = offset
        zero_boxes = []
        #sx, sy = float(new_w) / image_w, float(new_h) / image_h
        for obj in range(len(copy_objs)):
            copy_objs[obj]['xmin'] = int(self._constrain(0, target_w, copy_objs[obj]['xmin'] * padding_w /src_w + dx))
            copy_objs[obj]['xmax'] = int(self._constrain(0, target_w, copy_objs[obj]['xmax'] * padding_w / src_w + dx))
            copy_objs[obj]['ymin'] = int(self._constrain(0, target_h, copy_objs[obj]['ymin'] * padding_h / src_h + dy))
            copy_objs[obj]['ymax'] = int(self._constrain(0, target_h, copy_objs[obj]['ymax'] * padding_h / src_h + dy))

            # if copy_objs[obj]['xmin'] <= copy_objs[obj]['xmax'] or copy_objs[obj]['ymin'] <= copy_objs[obj]['ymax']:
            #     zero_boxes += [obj]
            #     continue

        #boxes = [copy_objs[i] for i in range(len(copy_objs)) if i not in zero_boxes]
        return image_data, copy_objs


    '''
    # code curtesy -> https://github.com/david8862/keras-YOLOv3-model-set/blob/master/common/data_utils.py
    # def letterbox_resize()
    '''
    def reshape_boxes(self,boxes, src_shape, target_shape, padding_shape, offset, horizontal_flip=False,
                      vertical_flip=False):
        """
        Reshape bounding boxes from src_shape image to target_shape image,
        usually for training data preprocess
        # Arguments
            boxes: Ground truth object bounding boxes,
                numpy array of shape (num_boxes, 5),
                box format (xmin, ymin, xmax, ymax, cls_id).
            src_shape: origin image shape,
                tuple of format (width, height).
            target_shape: target image shape,
                tuple of format (width, height).
            padding_shape: padding image shape,
                tuple of format (width, height).
            offset: top-left offset when padding target image.
                tuple of format (dx, dy).
            horizontal_flip: whether to do horizontal flip.
                boolean flag.
            vertical_flip: whether to do vertical flip.
                boolean flag.
        # Returns
            boxes: reshaped bounding box numpy array
        """
        if len(boxes) > 0:
            src_w, src_h = src_shape
            target_w, target_h = target_shape
            padding_w, padding_h = padding_shape
            dx, dy = offset

            # shuffle and reshape boxes
            np.random.shuffle(boxes)
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * padding_w / src_w + dx
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * padding_h / src_h + dy


            # # horizontal flip boxes if needed
            # if horizontal_flip:
            #     boxes[:, [0, 2]] = target_w - boxes[:, [2, 0]]
            # # vertical flip boxes if needed
            # if vertical_flip:
            #     boxes[:, [1, 3]] = target_h - boxes[:, [3, 1]]

            # check box coordinate range
            boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
            boxes[:, 2][boxes[:, 2] > target_w] = target_w
            boxes[:, 3][boxes[:, 3] > target_h] = target_h

            # check box width and height to discard invalid box
            boxes_w = boxes[:, 2] - boxes[:, 0]
            boxes_h = boxes[:, 3] - boxes[:, 1]
            boxes = boxes[np.logical_and(boxes_w > 1, boxes_h > 1)]  # discard invalid box

        return boxes

    def letterbox_resize(self, image, target_shape, return_padding_info=False):
        """
        Resize image with unchanged aspect ratio using padding
        # Arguments
            image: origin image to be resize
                PIL Image object containing image data
            target_size: target image size,
                tuple of format (width, height).
            return_padding_info: whether to return padding size & offset info
                Boolean flag to control return value
        # Returns
            new_image: resized PIL Image object.
            padding_size: padding image size (keep aspect ratio).
                will be used to reshape the ground truth bounding box
            offset: top-left offset in target image padding.
                will be used to reshape the ground truth bounding box
        """

        #src_w, src_h = image.size
        src_w, src_h = image.shape[0:2][::-1] # if ussing opencv
        target_w, target_h = target_shape

        # calculate padding scale and padding offset
        scale = min(target_w / src_w, target_h / src_h)
        padding_w = int(src_w * scale)
        padding_h = int(src_h * scale)
        padding_size = (padding_w, padding_h)

        dx = (target_w - padding_w) // 2
        dy = (target_h - padding_h) // 2
        offset = (dx, dy)

        # create letterbox resized image
        # image = image.resize(padding_size, Image.BICUBIC)
        # new_image = Image.new('RGB', target_shape, (128, 128, 128))
        # new_image.paste(image, offset)

        #using cv2
        image = cv2.resize(image, padding_size, interpolation=cv2.INTER_CUBIC)
        new_image = np.zeros((target_h, target_w, 3), np.uint8)
        new_image.fill(128)
        new_image[dy:dy + padding_h, dx:dx + padding_w, :] = image

        if return_padding_info:
            return new_image, padding_size, offset
        else:
            return new_image


    def _constrain(self,min_v, max_v, value):
        if value < min_v: return min_v
        if value > max_v: return max_v
        return value




