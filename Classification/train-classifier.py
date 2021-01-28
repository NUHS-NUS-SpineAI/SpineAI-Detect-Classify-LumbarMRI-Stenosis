import tensorflow as tf
import numpy as np
import itertools
import sys
import os

from tensorflow.keras import layers
from tensorflow.keras import backend as K

# for saving models and csvlogger
import datetime

print("TF version: ", tf.version.VERSION)

tf.keras.backend.clear_session()  # For easy reset of notebook state.

#  config = tf.ConfigProto()
#  config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#  sess = tf.Session(config=config)
#  K.set_session(sess)  # set this TensorFlow session as the default session for Keras

# dimensions of our images.
img_width, img_height = 150, 150
# IMAGE_SIZE    = (100, 100)
# CROP_LENGTH   = 84

if len(sys.argv) != 3:
    print('Error: pls provide train path and validation path')
    exit(0)


train_data_dir = sys.argv[1]
validation_data_dir = sys.argv[2]


nb_classes = 4
nb_train_samples = 0
nb_validation_samples = 0
nb_sample_per_class = []
nb_val_sample_per_class = []

folders = ['1', '2', '3', '4']
for folder in folders:
    num_tr = len(os.listdir(os.path.join(train_data_dir, folder)))
    nb_train_samples += num_tr
    nb_sample_per_class.append(num_tr)

for folder in folders:
    num_val = len(os.listdir(os.path.join(validation_data_dir, folder)))
    nb_validation_samples += num_val
    nb_val_sample_per_class.append(num_val)

# data_folder = train_data_dir.split(os.sep)
# data_folder = [e for e in data_folder if e != '']
# data_folder = data_folder[-2]
# print(data_folder)
print("\nnb_train_samples: ", nb_train_samples)
print("\nnb_validation_samples: ", nb_validation_samples)
print("\nnb_sample_per_class: ", nb_sample_per_class)
print("\nnb_val_sample_per_class: ", nb_val_sample_per_class)
print("--")

epochs = 100
batch_size = 128

from functools import partial, update_wrapper

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in itertools.product(range(nb_cl), range(nb_cl)):
        final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p] ,K.floatx())* K.cast(y_true[:, c_t],K.floatx()))
    return K.categorical_crossentropy(y_pred, y_true) * final_mask

w_array = np.ones((4,4))
w_array[0, 1] = 1
w_array[0, 2] = 1
w_array[0, 3] = 1
w_array[1, 0] = float(nb_sample_per_class[0])/float(nb_sample_per_class[1])
w_array[1, 2] = float(nb_sample_per_class[0])/float(nb_sample_per_class[1])
w_array[1, 3] = float(nb_sample_per_class[0])/float(nb_sample_per_class[1])
w_array[2, 0] = float(nb_sample_per_class[0])/float(nb_sample_per_class[2])
w_array[2, 1] = float(nb_sample_per_class[0])/float(nb_sample_per_class[2])
w_array[2, 3] = float(nb_sample_per_class[0])/float(nb_sample_per_class[2])
w_array[3, 0] = float(nb_sample_per_class[0])/float(nb_sample_per_class[3])
w_array[3, 1] = float(nb_sample_per_class[0])/float(nb_sample_per_class[3])
w_array[3, 2] = float(nb_sample_per_class[0])/float(nb_sample_per_class[3])

ncce = partial(w_categorical_crossentropy, weights=w_array)
ncce.__name__ = 'w_categorical_crossentropy'

# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


#  """ # RSNA version custom small model
model = tf.keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(img_height, img_width, 3), name='conv1'))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (3, 3), padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (3, 3), padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Dropout(0.2))

model.add(layers.Flatten())

model.add(layers.Dense(512))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(nb_classes, name='output'))
model.add(layers.Activation('softmax'))
#  """

# model.compile(loss=ncce,
#               optimizer=sgd,
#               metrics=['accuracy'])


# base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# # let's add a fully-connected layer
# x = Dense(1024, activation='relu')(x)
# # and a logistic layer -- let's say we have 200 classes
# predictions = Dense(4, activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=predictions)


model.compile(loss=ncce,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

# Display the model's architecture
model.summary()

# # random crop patch from:
# # https://mc.ai/extending-keras-imagedatagenerator-to-support-random-cropping/
# def random_crop(img, random_crop_size):
#     # Note: image_data_format is 'channel_last'
#     assert img.shape[2] == 3
#     height, width = img.shape[0], img.shape[1]
#     dy, dx = random_crop_size
#     x = np.random.randint(0, width - dx + 1)
#     y = np.random.randint(0, height - dy + 1)
#     return img[y:(y+dy), x:(x+dx), :]


# def crop_generator(batches, crop_length):
#     """Take as input a Keras ImageGen (Iterator) and generate random
#     crops from the image batches generated by the original iterator.
#     """
#     while True:
#         batch_x, batch_y = next(batches)
#         batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
#         for i in range(batch_x.shape[0]):
#             batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
#         yield (batch_crops, batch_y)


# this is the augmentation configuration we will use for training
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.1,
    #  horizontal_flip=True,
    rotation_range=5,
    #  width_shift_range=0.01,
    #  height_shift_range=0.01,
    #  brightness_range=[0.2, 1.0],
    )

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

# flow_from_directory will print:
# Found xxx images belonging to xxx classes
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# iterator that returns image_batch, label_batch pairs
for image_batch, label_batch in train_generator:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break
for image_batch, label_batch in validation_generator:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break


# # crop from 100 to CROP_LENGTH = 84
# train_crops = crop_generator(train_batches, CROP_LENGTH)

########################
# callbacks setting
########################
# for modelcheckpoint:
# if True, then only the model's weights will
# be saved (model.save_weights(filepath)),
# else the full model is saved (model.save(filepath)).

# for ModelCheckpoint and model saves
# Saving everything into a single archive in the TensorFlow SavedModel
# format (or in the older Keras H5 format). This is the standard practice.
formatted_time = datetime.datetime.now().strftime("%m%d-%H%M")
save_model_dir = "Axial_center_resnetscale150V1_150x150bat128_6LDropout_Date{}".format(formatted_time)

if not os.path.exists(save_model_dir):
    print(save_model_dir, " will be created")
    os.makedirs(save_model_dir)

# store the model json as:
store_model_json_name = "axial-center-Date{}.json".format(formatted_time)
# store model
model_json = model.to_json()
model_json_path = os.path.join(save_model_dir, store_model_json_name)
with open(model_json_path, "w") as json_file:
    json_file.write(model_json)

checkpoint_filepath = os.path.join(
    save_model_dir,
    "{0}_Ep{{epoch:02d}}_ValAcc{{val_acc:.3f}}_ValLoss{{val_loss:.2f}}.h5"
    .format(save_model_dir)
)

callbacks = [
    # tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss',
    #     patience=15,
    #     verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_acc',
        save_best_only=False,
        save_weights_only=True,
        verbose=1,
        save_freq="epoch"),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5, # usually 0.1
        patience=10,
        verbose=1),
    tf.keras.callbacks.CSVLogger(
        filename=os.path.join(
            save_model_dir,
            '{}.csv'.format(save_model_dir)
        ),
        append=False,
        separator=','),
]


train_steps = np.ceil(nb_train_samples / batch_size)
print("len train_generator: ", len(train_generator))

val_steps = np.ceil(nb_validation_samples / batch_size)
print("len validation_generator: ", len(validation_generator))

train_history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=val_steps,
    callbacks=callbacks)
