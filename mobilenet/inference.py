import os
import cv2
import numpy as np
import tensorflow as tf

import grad_cam
from load import Get_Test_Image


# グラボのメモリが不足してる時はこれをいれる
#グラボがないときはコメントアウト
physical_devices = tf.config.list_physical_devices('GPU')
gpu_id = [0] # 使用するGPUをリストで指定
if len(physical_devices) > 0:
    for i, device in enumerate(physical_devices):
        if i in gpu_id:
            tf.config.experimental.set_visible_devices(physical_devices[i], 'GPU')
            tf.config.experimental.set_memory_growth(device, True)
            print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")


classidx2classname = ["pug", "bulldog", "others"]

#設定
BATCH_SIZE = 32
EPOCH_NUM = 20
IMAGE_SIZE = (224, 224)

# データセット用意=
x_test, y_test, dataset_cate_name, filenames = Get_Test_Image()

# numpy --> tf.data
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# batch
test_dataset = test_dataset.batch(BATCH_SIZE)

######################NN構築(CNN)##############################
model = tf.keras.models.load_model("./model_save")
# model.summary()
######################NN構築##############################

# print('x_test:', x_test.shape)
predicts = model.predict(x_test)
# print(predicts)
# print(y_test)

for idx in range(len(x_test)):
    out_heatmaps = []
    for visualize_class in range(x_test.shape[3]):
        img = x_test[idx]
        uint8_BGRimg, heatmap, output_heatmapimg = grad_cam.grad_cam(model, img, 'out_relu', visualize_class)
        grad_cam.save_grad_cam_outputs(uint8_BGRimg, heatmap, output_heatmapimg, idx, classidx2classname[visualize_class])
        out_heatmaps.append(output_heatmapimg)
    grad_cam.save_grad_cam_outputs_to_oneimg(uint8_BGRimg, out_heatmaps, idx, x_test.shape[3], predicts[idx], y_test[idx], classidx2classname)

# print("dataset_cate_name")
# print(dataset_cate_name)

print(filenames)



#モデル評価
score = model.evaluate(x_test, y_test, verbose=2)
print('損失:', score[0])
print('正答率:', score[1])
