import os
import cv2
import numpy as np
import tensorflow as tf

import grad_cam
from load import Get_Training_and_Test_Image

classidx2classname = ["pug", "bull", "others"]

# グラボのメモリが不足してる時はこれをいれる
#グラボがないときはコメントアウト
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

#設定
BATCH_SIZE = 32
EPOCH_NUM = 20
IMAGE_SIZE = (224, 224)

# データセット用意=
_, _, x_test, y_test, dataset_cate_name = Get_Training_and_Test_Image()

# numpy --> tf.data
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# batch
test_dataset = test_dataset.batch(BATCH_SIZE)

######################NN構築(CNN)##############################
model = tf.keras.models.load_model("./model_save")
model.summary()
######################NN構築##############################

print('x_test:', x_test.shape)
for visualise_class in range(x_test.shape[3]):
    for idx in range(len(x_test)):
        img = x_test[idx]
        uint8_BGRimg, heatmap, output_image = grad_cam.grad_cam(model, img, 'out_relu', idx, visualise_class)
        grad_cam.save_grad_cam_outputs(uint8_BGRimg, heatmap, output_image, idx, classidx2classname[visualise_class])

#評価
score = model.evaluate(x_test, y_test, verbose=2)
print('損失:', score[0])
print('正答率:', score[1])
