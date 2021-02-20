import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2
import seaborn as sb

def sigmoid(x):
	return 1. / (1. + np.exp(-x))

def softmax(x, axis=-1):
    x = x - np.amax(x, axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / e_x.sum(axis, keepdims=True)

"""
Modifies the input image to desired network height and width
"""
def load_image_and_convert_scaled_pixels(filename, net_w, net_h):
    # load image
    image = load_img(filename)
    width, height = image.size

    # load the image with req shape
    image = load_img(filename, target_size=(net_w, net_h))
    image = img_to_array(image)
    image = image.astype('float32')
    image /= 255.0
    image = np.expand_dims(image, axis=0)

    return image, width, height

def normalize(image):
    return image/255.


def letterbox_resize(image, target_shape, return_padding_info=False):
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


def plot_bboxes(img, bboxes, color_map):
    '''
        Plot the  bounding boxes of a given image with a pre-defined colormap
    '''
    show_img = np.copy(img)
    #fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontFace = cv2.FONT_HERSHEY_TRIPLEX
    scale = 0.5
    thickness = 1

    for bbox in bboxes:
        label = bbox['class']
        pts = bbox['bbox']  # its a tuple of (x,y, x+height, y+width)

        pt1 = (int(pts[0]), int(pts[1]))  # (x,y)
        pt2 = (int(pts[2]), int(pts[3]))  # (x+width, y+height)

        cv2.rectangle(show_img, pt1, pt2, color_map[label], 2)

        textSize, baseline = cv2.getTextSize(label, fontFace=fontFace,
                                             fontScale=scale,
                                             thickness=thickness)

        cv2.rectangle(show_img, pt1, (pt1[0]+textSize[0], pt1[1]+textSize[1]),
                      color_map[label])

        cv2.putText(show_img, label, (pt1[0], pt1[1]+baseline*2),
                    fontFace, scale, (255, 255, 255), thickness)

    return show_img

def make_color_map(classes):
    '''
        Create a color map for each class
    '''
    names = sorted(set(classes))
    n = len(names)
    cp = sb.color_palette("Paired", n)
    cp[:] = [tuple(int(255*c) for c in rgb) for rgb in cp]

    return dict(zip(names, cp))


