import pickle
import sys
import os
from shutil import rmtree

from tensorflow.python.keras.callbacks import EarlyStopping

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
base_path = '/mywork/PGDDS-IIITB/MyPractice'
proj_dir_path = os.path.join(base_path,"MyMScProj")
dataset_base_path = '/mywork/PGDDS-IIITB/MyDatasets'
sys.path.append(base_path)
sys.path.append(proj_dir_path)

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD

from MyMScProj.jmod.keypoint.centernet.generators.miotcd import BatchGenerator
from MyMScProj.jmod.keypoint.centernet.models.resnet import centernet
from MyMScProj.jmod.preprocess import Preprocess

#backend_weight_path=os.path.join(proj_dir_path, 'jmod/keypoint/centernet/ResNet-50-model.keras.h5')
backend_weight_path=os.path.join(proj_dir_path, 'jmod/keypoint/centernet/ResNet-101-model.keras.h5')
tensorboard_dir=os.path.join(proj_dir_path,"tensorboard/centernet_tensorboard_logs_tst7")
saved_weights_dir=os.path.join(proj_dir_path,"jmod/keypoint/centernet/jm_centernet_tst6")
prev_saved_weights_name = "miotcd_22_7004.0000_7327.9639.h5"
labels = ['articulated_truck', 'bicycle', 'bus', 'car',
           'motorcycle', 'motorized_vehicle', 'non-motorized_vehicle',
           'pedestrian', 'pickup_truck', 'single_unit_truck', 'work_van']
annot_csv_file = os.path.join(dataset_base_path,"/MIO-TCD/MIO-TCD-Localization/gt_train.csv")
train_img_dir = os.path.join(dataset_base_path,"/MIO-TCD/MIO-TCD-Localization/train")

# training params
batch_size=32
input_size=512
freeze_bn=False
epochs=50
steps_per_epoch=20
max_queue_size=8
workers=4
max_objects=20

with open(os.path.join(proj_dir_path,"all_insts"), "rb") as rh:
    train_ds = pickle.load(rh)
    rh.close()

with open(os.path.join(proj_dir_path,"seen_labels"), "rb") as rh:
    seen_train_labels = pickle.load(rh)
    rh.close()

def dummy_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(y_pred))

def create_callbacks(training_model, prediction_model, validation_generator):
    """
    Creates the callbacks to use during training.

    Args
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []
    if os.path.exists(tensorboard_dir):
        rmtree(tensorboard_dir)
        os.makedirs(tensorboard_dir)
    tensorboard_callback = None

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=0,
            batch_size=batch_size,
            write_graph=True,
            write_grads=True,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None
    )
    callbacks.append(tensorboard_callback)

    # from MyMScProj.jmod.keypoint.centernet.eval.pascal import Evaluate
    # evaluation = Evaluate(validation_generator, prediction_model, tensorboard=tensorboard_callback)
    # callbacks.append(evaluation)

    # save the model
    if not os.path.exists(saved_weights_dir):
        os.makedirs(saved_weights_dir)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(
                saved_weights_dir,
                'miotcd_{epoch:02d}_{loss:.4f}_{val_loss:.4f}.h5'),
            verbose=1,
            save_best_only=True,
            #monitor="mAP",
            mode='min',
            monitor="loss",

        )
    callbacks.append(checkpoint)

    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=2,
        verbose=1,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=0
    ))

    early_stop = EarlyStopping(
        monitor='loss',
        min_delta=0.01,
        patience=7,
        mode='min',
        verbose=1
    )
    callbacks.append(early_stop)

    return callbacks

# create training instances
preproc = Preprocess(annot_csv_file, train_img_dir, labels)
print(">>>> Create training instances <<<<")
train_insts, valid_insts, labels, max_box_per_image \
    = preproc.create_training_instances(train_ds, seen_train_labels,sampling=30000)


common_args = {
    'batch_size': batch_size,
    'input_size': input_size,
    'labels': labels
    }
train_generator = BatchGenerator(train_insts,base_dir=train_img_dir,**common_args)
valid_generator = BatchGenerator(valid_insts,base_dir=train_img_dir,**common_args)

num_classes = train_generator.num_classes()
model, prediction_model, debug_model, loss_dict = centernet(num_classes=num_classes, backbone="resnet101",
                                                            input_size=input_size, max_objects=max_objects,
                                                            freeze_bn=freeze_bn)

# create the model
print('Loading model, this may take a second...')
# model.load_weights(backend_weight_path,
#                        by_name=True, skip_mismatch=True)
if not os.path.exists(os.path.join(saved_weights_dir, prev_saved_weights_name)):
    model.load_weights(backend_weight_path,
                       by_name=True, skip_mismatch=True)
else:
    model.load_weights(os.path.join(saved_weights_dir, prev_saved_weights_name),
                       by_name=True, skip_mismatch=False)

# freeze layers
if freeze_bn:
   for i in range(190):
   # for i in range(175):
    model.layers[i].trainable = False

# add metric for three loss heads hm_loss, wh_loss, reg_loss
for (name, metric) in loss_dict.items():
    model.add_metric(metric, name=name, aggregation='mode')

# compile model
#optimizer = Adam(lr=1e-4)
optimizer=SGD(lr=1e-5, momentum=0.9, nesterov=True, decay=1e-5)
model.compile(optimizer=optimizer, loss={'centernet_loss': lambda y_true, y_pred: y_pred})
# model.compile(optimizer=optimizer, loss={'centernet_loss': lambda y_true, y_pred: y_pred},
#               run_eagerly=True, metrics=['acc'])
# model.compile(optimizer=SGD(lr=1e-5, momentum=0.9, nesterov=True, decay=1e-5),
#                loss={'centernet_loss': lambda y_true, y_pred: y_pred})

# print model summary
print(model.summary())

# create the callbacks
callbacks = create_callbacks( model, prediction_model, valid_generator )

    # if not args.compute_val_loss:
    #     validation_generator = None

# start training
model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps_per_epoch,
        initial_epoch=0,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        workers=workers,
        use_multiprocessing=True,
        max_queue_size=max_queue_size,
        validation_data=valid_generator,
        validation_steps = steps_per_epoch//2
)










####


