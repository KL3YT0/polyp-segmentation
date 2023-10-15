import numpy as np
import os
import tensorflow as tf
from keras.models import Model
from keras.layers import (
    UpSampling2D,
    Conv2D,
    MaxPooling2D,
    Input,
    Concatenate,
)
from keras.applications.vgg16 import VGG16
from tensorflow.keras import backend as K
import albumentations as A
import numpy as np
import random
from dotenv import load_dotenv, dotenv_values

load_dotenv()

BATH_SIZE = 4

transform = A.Compose([
    A.Rotate(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.2),
    A.HorizontalFlip(p=0.2),
])

train_images_path = os.path.join(os.getenv('GOOGLE_DRIVE_URL'), "train_images")
train_masks_path = os.path.join(os.getenv('GOOGLE_DRIVE_URL'), "train_masks")
val_images_path = os.path.join(os.getenv('GOOGLE_DRIVE_URL'), "val_images")
val_masks_path = os.path.join(os.getenv('GOOGLE_DRIVE_URL'), "val_masks")

data_gen_args = dict()

image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **data_gen_args, preprocessing_function=lambda x: x / 255.0
)
mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **data_gen_args,
    preprocessing_function=lambda x: np.where(x > 0, 1, 0).astype(x.dtype)
)

val_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **data_gen_args, preprocessing_function=lambda x: x / 255.0
)
val_mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **data_gen_args,
    preprocessing_function=lambda x: np.where(x > 0, 1, 0).astype(x.dtype)
)

train_images = image_datagen.flow_from_directory(
    train_images_path,
    color_mode="rgb",
    batch_size=BATH_SIZE,
    target_size=(256, 256),
    shuffle=False,
)

train_masks = mask_datagen.flow_from_directory(
    train_masks_path,
    class_mode="binary",
    color_mode="grayscale",
    target_size=(256, 256),
    batch_size=BATH_SIZE,
    shuffle=False,
)

val_images = val_image_datagen.flow_from_directory(
    val_images_path,
    color_mode="rgb",
    batch_size=BATH_SIZE,
    target_size=(256, 256),
    shuffle=False,
)

val_masks = val_mask_datagen.flow_from_directory(
    val_masks_path,
    class_mode="binary",
    color_mode="grayscale",
    target_size=(256, 256),
    batch_size=BATH_SIZE,
    shuffle=False,
)


def train_generator(a, b):
    while True:
        next_a = next(a)[0]
        next_b = next(b)[0]
        rndIndex = int(random.random() * BATH_SIZE - 1)

        transformed = transform(image=next_a[rndIndex], mask=next_b[rndIndex])
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']

        next_a[rndIndex] = transformed_image
        next_b[rndIndex] = transformed_mask

        yield next_a, next_b

def val_generator(a, b):
    while True:
        yield next(a)[0], next(b)[0]

base_model = VGG16(weights="imagenet", input_shape=(256, 256, 3), include_top=False)

for layer in base_model.layers:
            layer.trainable = False

pool_1 = base_model.get_layer("block1_pool").output
pool_2 = base_model.get_layer("block2_pool").output
conv_1_2 = base_model.get_layer("block1_conv2").output
# pool_3 = base_model.get_layer("block3_pool").output
# pool_4 = base_model.get_layer("block4_pool").output
# conv_2_2 = base_model.get_layer("block2_conv2").output
# conv_3_2 = base_model.get_layer("block3_conv2").output
# conv_4_2 = base_model.get_layer("block4_conv2").output
# conv_5_2 = base_model.get_layer("block5_conv3").output

# inputs = Input(shape=(256, 256, 3))

# conv_1_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(inputs)
# conv_1_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(conv_1_1)
# pool_1 = MaxPooling2D(2)(conv_1_2)


conv_2_1 = Conv2D(64, (3, 3), activation='relu', padding="same")(pool_1)
conv_2_2 = Conv2D(64, (3, 3), activation='relu', padding="same")(conv_2_1)
pool_2 = MaxPooling2D(2)(conv_2_2)


conv_3_1 = Conv2D(128, (3, 3), activation='relu', padding="same")(pool_2)
conv_3_2 = Conv2D(128, (3, 3), activation='relu', padding="same")(conv_3_1)
pool_3 = MaxPooling2D(2)(conv_3_2)

conv_4_1 = Conv2D(256, (3, 3), activation='relu', padding="same")(pool_3)
conv_4_2 = Conv2D(256, (3, 3), activation='relu', padding="same")(conv_4_1)
pool_4 = MaxPooling2D(2)(conv_4_2)

conv_5_1 = Conv2D(512, (3, 3), activation='relu', padding="same")(pool_4)
conv_5_2 = Conv2D(512, (3, 3), activation='relu', padding="same")(conv_5_1)

############################################################

up_6 = UpSampling2D(2, interpolation="bilinear")(conv_5_2)
conc_6 = Concatenate()([up_6, conv_4_2]) # Может стоит брать pool
conv_6_1 = Conv2D(256, (3, 3), activation='relu', padding="same")(conc_6)
conv_6_2 = Conv2D(256, (3, 3), activation='relu', padding="same")(conv_6_1)

up_7 = UpSampling2D(2, interpolation="bilinear")(conv_6_2)
conc_7 = Concatenate()([up_7, conv_3_2])
conv_7_1 = Conv2D(128, (3, 3), activation='relu', padding="same")(conc_7)
conv_7_2 = Conv2D(128, (3, 3), activation='relu', padding="same")(conv_7_1)

up_8 = UpSampling2D(2, interpolation="bilinear")(conv_7_2)
conc_8 = Concatenate()([up_8, conv_2_2])
conv_8_1 = Conv2D(64, (3, 3), activation='relu', padding="same")(conc_8)
conv_8_2 = Conv2D(64, (3, 3), activation='relu', padding="same")(conv_8_1)

up_9 = UpSampling2D(2, interpolation="bilinear")(conv_8_2)
conc_9 = Concatenate()([up_9, conv_1_2])
conv_9_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(conc_9)
conv_9_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(conv_9_1)

outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv_9_2)

model = Model(inputs=base_model.inputs, outputs=[outputs])

binary_accuracy = tf.keras.metrics.BinaryAccuracy(
    name="binary_accuracy", dtype=None, threshold=0.5
)

recall = tf.keras.metrics.Recall()

mean_iou = tf.keras.metrics.MeanIoU(num_classes=2)

def dice_coef(y_true, y_pred):
   return 1. - dice_loss(y_true=y_true, y_pred=y_pred)

def dice_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f* y_pred_f)
    val = (2. * intersection + K.epsilon()) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + K.epsilon())
    return 1. - val

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=dice_loss, metrics=[binary_accuracy, recall, dice_coef, mean_iou])

model.fit(
    train_generator(train_images, train_masks),
    validation_data=val_generator(val_images, val_masks),
    validation_steps=5,
    steps_per_epoch=100,
    epochs=60,
    verbose=1,
    class_weight=None,
    use_multiprocessing=False,
    shuffle=True,
    initial_epoch=0,
    workers=1,
)