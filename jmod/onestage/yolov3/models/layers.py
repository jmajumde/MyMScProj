from functools import reduce, wraps

import tensorflow as tf
from tensorflow.keras.layers import Add, BatchNormalization, LeakyReLU, Conv2D, ZeroPadding2D, UpSampling2D
from tensorflow.keras.layers import Concatenate
from keras.layers.merge import add
from tensorflow.keras.regularizers import l2

L2_FACTOR = 1e-5

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by CustomBatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)

    interim_model = compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(epsilon=0.001, trainable=False),
        LeakyReLU(alpha=0.1 ))

    return interim_model


def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for YoloConv2D."""
    #darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    #darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs = {'padding': 'valid' if kwargs.get('strides')==(2,2) else 'same'}
    darknet_conv_kwargs.update(kwargs)
    return YoloConv2D(*args, **darknet_conv_kwargs)


@wraps(Conv2D)
def YoloConv2D(*args, **kwargs):
    """Wrapper to set Yolo parameters for Conv2D."""
    yolo_conv_kwargs = {'kernel_regularizer': l2(L2_FACTOR)}
    yolo_conv_kwargs['bias_regularizer'] = l2(L2_FACTOR)
    yolo_conv_kwargs.update(kwargs)
    #yolo_conv_kwargs = kwargs
    return Conv2D(*args, **yolo_conv_kwargs)


def CustomBatchNormalization(*args, **kwargs):
    if tf.__version__ >= '2.2':
        from tensorflow.keras.layers.experimental import SyncBatchNormalization
        BatchNorm = SyncBatchNormalization
    else:
        BatchNorm = BatchNormalization

    return BatchNorm(*args, **kwargs)

def yolo3_predictions(feature_maps, feature_channel_nums, num_classes):
    f13, f26, f52 = feature_maps
    f13_channels, f26_channels, f52_channels = feature_channel_nums

    # feature map 1 head & output (13x13 for 416 input) - starting with 1024 filters
    x, y1 = make_last_layers(f13, f13_channels, 3 * (num_classes + 5), predict_id='1')
    # upsample fpn merge for feature maps 1 and 2
    x = compose(DarknetConv2D_BN_Leaky(f26_channels//2, (1,1)),
                UpSampling2D(2))(x)
    x = Concatenate()([x,f26])

    # feature map 2 head & output (26x26 for 416 input) - starting with 512 filters
    x, y2 = make_last_layers(f26, f26_channels, 3 * (num_classes + 5), predict_id='2')
    # upsample fpn merge for feature maps 2 and 3
    x = compose(DarknetConv2D_BN_Leaky(f52_channels//2, (1, 1)),
                UpSampling2D(2))(x)
    x = Concatenate()([x, f52])

    # feature map 3 head & output (52x52 for 416 input) - starting with 128 filters
    x, y3 = make_last_layers(f52, f52_channels//2, 3 * (num_classes + 5), predict_id='3')

    return y1, y2, y3


def make_last_layers(x, num_filters, out_filters, predict_filters=None, predict_id='1'):
    '''
    Following the pred_yolo1, pred_yolo2 and pred_yolo3 as per exriencor code
     https://github.com/experiencor/keras-yolo3
    '''
    # if predict_id == '1' or predict_id == '2':
    #     # Conv2D_BN_Leaky layers followed by a Conv2D_linear layer
    #     y = compose(
    #         DarknetConv2D_BN_Leaky(num_filters, (3, 3)),
    #         DarknetConv2D(out_filters, (1, 1), name='predict_conv_' + predict_id))(x)
    #
    # if predict_id == '3':
    # 6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer
    # num_filters here 128
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)

    if predict_filters is None:
        predict_filters = num_filters * 2

    y = compose(
        DarknetConv2D_BN_Leaky(predict_filters, (3, 3)),
        DarknetConv2D(out_filters, (1, 1), name='predict_conv_' + predict_id))(x)

    return x, y

def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for YoloConv2D."""
    # darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    # darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs = {'padding': 'valid' if kwargs.get('strides') == (2, 2) else 'same'}
    darknet_conv_kwargs.update(kwargs)
    return YoloConv2D(*args, **darknet_conv_kwargs)

@wraps(Conv2D)
def YoloConv2D(*args, **kwargs):
    """Wrapper to set Yolo parameters for Conv2D."""
    yolo_conv_kwargs = {'kernel_regularizer': l2(L2_FACTOR)}
    yolo_conv_kwargs['bias_regularizer'] = l2(L2_FACTOR)
    yolo_conv_kwargs.update(kwargs)
    #yolo_conv_kwargs = kwargs
    return Conv2D(*args, **yolo_conv_kwargs)


def conv_block(inp, convs, do_skip=True):
    x = inp
    count = 0

    for conv in convs:
        if count == (len(convs) - 2) and do_skip:
            skip_connection = x
        count += 1

        if conv['stride'] > 1: x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # unlike tensorflow darknet prefer left and top paddings
        x = Conv2D(conv['filter'],
                   conv['kernel'],
                   strides=conv['stride'],
                   padding='valid' if conv['stride'] > 1 else 'same',
                   # unlike tensorflow darknet prefer left and top paddings
                   name='conv_' + str(conv['layer_idx']),
                   use_bias=False if conv['bnorm'] else True)(x)
        if conv['bnorm']: x = BatchNormalization(epsilon=0.001, trainable=False,
                                                 name='bnorm_' + str(conv['layer_idx']))(x)
        if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)

    return add([skip_connection, x]) if do_skip else x

























#####