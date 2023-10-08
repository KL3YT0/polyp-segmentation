import numpy as np
import os
import tensorflow as tf
from keras.models import Model
from keras.layers import (
    UpSampling2D,
    Conv2D,
    Activation,
    MaxPooling2D,
    Input,
    Concatenate,
)
from keras.applications.vgg16 import VGG16
from dotenv import load_dotenv, dotenv_values

load_dotenv()

BATH_SIZE = 16

"""
Структура директории датасета:
train-images/
...train/
......train_1.jpg
......train_2.jpg
train-masks/
...train/
......train_1.jpg
......train_2.jpg
val-images/
...val/
......val_1.jpg
......val_2.jpg
val-masks/
...val/
......val_1.jpg
......val_2.jpg
"""

train_images_path = os.path.join(os.getenv('GOOGLE_DRIVE_URL'), "train_images")
train_masks_path = os.path.join(os.getenv('GOOGLE_DRIVE_URL'), "train_masks")

data_gen_args = dict()

image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **data_gen_args, preprocessing_function=lambda x: x / 255.0
)
mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
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


def my_image_mask_generator(image_data_generator, mask_data_generator):
    while True:
        yield next(image_data_generator)[0], next(mask_data_generator)[0]

inputs = Input(shape=(256, 256, 3))

conv_1_1 = Conv2D(32, (3, 3), activation='relu', padding="same")(inputs)
conv_1_2 = Conv2D(32, (3, 3), activation='relu', padding="same")(conv_1_1)
pool_1 = MaxPooling2D(2)(conv_1_2)


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

model = Model(inputs=inputs, outputs=[outputs])

b_acc = tf.keras.metrics.BinaryAccuracy(
    name="binary_accuracy", dtype=None, threshold=0.5
)

rec = tf.keras.metrics.Recall()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[b_acc, rec])

model.fit_generator(
    my_image_mask_generator(train_images, train_masks),
    steps_per_epoch=100,
    epochs=10,
    verbose=1,
    class_weight=None,
    use_multiprocessing=False,
    shuffle=True,
    initial_epoch=0,
    workers=1,
)
