#返り値として訓練用データセットが返る関数

#訓練データ
#x_train[画像枚数][高さ][幅][RGB]            入力
#y_train[画像枚数][クラス数]                 出力

#テストデータ
#x_test
#y_test
import glob
import os
import random

import cv2
from PIL import Image
import numpy as np


########################################################################
###メイン処理, 要求された画像数のx_train, y_trainm, x_test, y_testを返す###
########################################################################
def Get_Training_and_Test_Image():

    #データセット内のデータセット群を取得
    dataset_folders_path = Get_dataset_name()

    #カテゴリ名取得
    dataset_cate_name = Get_dataset_name()

    #画像データサイズを決定(このサイズにリサイズする)
    img_size=(224, 224)

    #データから入力行列, 出力行列を作成(訓練用)
    x_train, y_train = Get_xy_data(dataset_folders_path, dataset_cate_name, img_size, 0)

    #データから入力行列, 出力行列を作成(評価用)
    x_test, y_test = Get_xy_data(dataset_folders_path, dataset_cate_name, img_size, 1)

    print(type(x_train), x_train.shape)
    print(type(y_train), y_train.shape)
    print(type(x_test), x_test.shape)
    print(type(y_test), y_test.shape)

    return x_train, y_train, x_test, y_test, dataset_cate_name


#################################
#データセットのサブフォルダ群を取得#
#################################
def Get_dataset_name():

    #カレントディレクトリ直下のdatasetフォルダを探索
    dataset_path = os.getcwd() + "\dataset"

    #サブフォルダーのパスを取得
    folders = os.listdir(dataset_path)
    folders_name = [f for f in folders if os.path.isdir(os.path.join(dataset_path, f))]

    return folders_name

#########################
#x_data, y_dataを返す関数#
#########################
def Get_xy_data(dataset_folders_path, dataset_cate_name, img_size, mode):

    #行列用意
    x_mats, y_mats = [], []

    #x_dataのパス用意
    x_data_paths = ["temp" for i in range(len(dataset_folders_path))]

    if(mode == 0):
        #訓練データの行列を取得するモード
        x_data_paths = ["dataset/" + dataset_folders_path[i] + "/train" for i in range(len(dataset_folders_path))]
    elif(mode == 1):
        #評価データの行列を取得するモード
        x_data_paths = ["dataset/" + dataset_folders_path[i] + "/test" for i in range(len(dataset_folders_path))]
    print(x_data_paths)

    #全サブフォルダを処理
    for folder_cnt in range(len(x_data_paths)):

        #フォルダ内のファイル取得
        img_file_names = os.listdir(x_data_paths[folder_cnt])

        #フォルダ内の画像を処理
        for img_file_cnt in range(len(img_file_names)):

            #画像の相対パス取得
            img_file_path = x_data_paths[folder_cnt] + "/" + img_file_names[img_file_cnt]

            #画像の前処理
            image = preparation_img(img_file_path, img_size)

            #画像をリストに追加
            x_mats.append(image)

            #正解値をリストに追加(サブフォルダの番号がそのままカテゴリ番号)
            y_mat = []
            for i in range(len(x_data_paths)):
                #フォルダカウントの要素に1をいれる
                if(i == folder_cnt):
                    y_mat.append(1)
                else:
                    y_mat.append(0)
            #今回の画像の正解値をy_matsに格納
            y_mats.append(y_mat)


    x_mats = np.asarray(x_mats, dtype=np.float32)
    y_mats = np.asarray(y_mats, dtype=np.uint8)

    #np.savetxt(img_file_names[img_file_cnt] + "_x.csv" ,x_mats[0,:,:,0], delimiter=',')

    return x_mats, y_mats

#########################
########画像前準備########
#########################
def preparation_img(img_file_path, img_size):

    #ファイルを開く
    image = Image.open(img_file_path)

    #正方形へ変換
    image = crop_to_square(image)

    #リサイズ
    image = image.resize(img_size)

    #RGBA→RGB
    if image.mode == "RGBA":
        image = image.convert("RGB")

    #numpy配列にして小数化
    image = np.asarray(image)
    image = image / 255.0

    return image


#############################################
########画像を正方形に変換する処理(借用)########
#############################################
def crop_to_square(image):
    size = min(image.size)
    left, upper = (image.width - size) // 2, (image.height - size) // 2
    right, bottom = (image.width + size) // 2, (image.height + size) // 2
    return image.crop((left, upper, right, bottom))

Get_Training_and_Test_Image()