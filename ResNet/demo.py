from PIL import Image
import numpy as np
import glob
import os
import cv2
import tensorflow as tf
from load import Get_dataset_name, preparation_img

img_size = (224, 224)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

###########################################
###任意の画像に対して推論→可視化のメイン関数###
###########################################
def demo_main():

    #NNの情報入手
    #同じモデルを読み込んで、重みやオプティマイザーを含むモデル全体を再作成
    model = tf.keras.models.load_model('model_img_recog_ramen_pan_bn.h5')
    model.summary()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    #画像のフォルダパス取得
    img_dir_path ="demo_img"
    img_file_names = os.listdir(img_dir_path)

    #フォルダ内の全ての画像で推論
    for img_cnt in range(len(img_file_names)):

        #画像パス取得
        img_path = img_dir_path + "/" + img_file_names[img_cnt]

        #画像前処理
        input_img = preparation_img(img_path, img_size)
        input_img = np.asarray(input_img, dtype=np.float32)
        input_img = np.expand_dims(input_img, axis = 0)

        #推論実施
        predict_results = model.predict(input_img)

        #カテゴリ名取得
        folders_name = Get_dataset_name()

        #推論結果取得
        predict_result = predict_results[0,:]

        #結果出力
        np.set_printoptions(precision=3, suppress=True)
        print("-------------------------------------")
        print(img_file_names[img_cnt])
        print("-------------------------------------")
        for category_cnt in range(len(folders_name)):
            print (folders_name[category_cnt] , ":  " , predict_result[category_cnt]*100, "%")

demo_main()