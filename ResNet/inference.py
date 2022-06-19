import os
import cv2
import numpy as np
import tensorflow as tf
from load import Get_Training_and_Test_Image

from grad_cam import grad_cam

# グラボのメモリが不足してる時はこれをいれる
#グラボがないときはコメントアウト
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

#epoch数とbatchサイズ設定
set_epoch_num = 20
set_batch_size = 64

#データセット用意=
x_train, y_train, x_test, y_test, dataset_cate_name = Get_Training_and_Test_Image()

# numpy --> tf.data
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# shuffle
# batch
test_dataset = test_dataset.batch(set_batch_size)

######################NN構築(CNN)##############################
input_shape = (224, 224, 3)

model = tf.keras.models.load_model("../trained_models/model_img_recog_pug_bull_bn")

model.summary()

######################NN構築##############################

for idx in range(len(x_test)):

    img = x_test[idx]


    grad_cam(model, img, 're_lu_35', idx)

#評価
score = model.evaluate(x_test, y_test, verbose=2)
print('損失:', score[0])
print('正答率:', score[1])
