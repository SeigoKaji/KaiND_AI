import os
import cv2
import numpy as np
import tensorflow as tf

import grad_cam
from load import Get_Test_Image

########################################おまじない#######################################
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
########################################おまじない#######################################

#結果格納配列の用意
result_list = [[0 for j in range(0, 3)] for i in range(0, 3)]
total = 0

#クラス設定
classidx2classname = ["00_pug", "01_bulldog", "02_others"]

# データセット用意
print("################データセット読み込み開始################")
x_test, y_test, dataset_cate_name, filenames = Get_Test_Image()

#モデル読み込み
print("################モデルの読み込み開始################")
model = tf.keras.models.load_model("./model_save")

#推論
print("################推論開始################")
predicts = model.predict(x_test)

#評価画像ループ
for idx in range(len(x_test)):
    out_heatmaps = []

    #正解と識別結果取得
    gt_class_idx = np.argmax(y_test[idx])
    recog_class_idx = np.argmax(predicts[idx])
    str_gt_class = classidx2classname[gt_class_idx]
    str_recog_class = classidx2classname[recog_class_idx]
    max_score = max(predicts[idx])
    
    print("######", idx + 1, "枚目######")
    print("ファイル名:    ", filenames[idx])
    print("正解クラス     :", str_gt_class)
    print("パグスコア     : ", predicts[idx][0])
    print("ブルドッグスコア: ", predicts[idx][1])
    print("Otherスコア    :", predicts[idx][2])
    print("最終結果       :", str_gt_class, "⇒", str_recog_class)
    print("最大スコア     :", max_score)

    #識別結果の加算
    result_list[gt_class_idx][recog_class_idx] = result_list[gt_class_idx][recog_class_idx] + 1
    total = total + 1

    #間違っている画像もしくは断言できていない画像のみヒートマップ作製
    if gt_class_idx != recog_class_idx or max_score < 0.8:
        #クラスループで各ヒートマップ作製
        for visualize_class in range(x_test.shape[3]):
            img = x_test[idx]
            uint8_BGRimg, heatmap, output_heatmapimg = grad_cam.grad_cam(model, img, 'out_relu', visualize_class)
            grad_cam.save_grad_cam_outputs(uint8_BGRimg, heatmap, output_heatmapimg, idx, str_gt_class)
            out_heatmaps.append(output_heatmapimg)

        #Gradcam画像の保存
        grad_cam.save_grad_cam_outputs_to_oneimg(uint8_BGRimg, out_heatmaps, idx, x_test.shape[3], predicts[idx], y_test[idx], classidx2classname)


#結果表示
print("#########################################")
print("#########################################")
print("パグ⇒パグ : ", result_list[0][0], "/", total)
print("パグ⇒ブル : ", result_list[0][1], "/", total)
print("パグ⇒Other: ", result_list[0][2], "/", total)
print("ブル⇒パグ : ", result_list[1][0], "/", total)
print("ブル⇒ブル : ", result_list[1][1], "/", total)
print("ブル⇒Other: ", result_list[1][2], "/", total)
print("Other⇒パグ : ", result_list[2][0], "/", total)
print("Other⇒ブル : ", result_list[2][1], "/", total)
print("Other⇒Other: ", result_list[2][2], "/", total)
print("#########################################")
print("#########################################")