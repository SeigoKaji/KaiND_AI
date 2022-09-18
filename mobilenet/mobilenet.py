import datetime
import math
import tensorflow as tf
from load import Get_Training_and_Test_Image


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


#設定
BATCH_SIZE = 32
EPOCH_NUM = 20
IMAGE_SIZE = (224, 224)
IMG_SHAPE = IMAGE_SIZE + (3,)

#学習データ取得
data_dir = "dataset"

x_train, y_train, x_test, y_test, dataset_cate_name = Get_Training_and_Test_Image()

# numpy --> tf.data
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# print(train_dataset)
# shuffle
# batch
train_dataset = train_dataset.shuffle(x_train.shape[0]).batch(BATCH_SIZE)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# backboneリード
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
base_model.trainable = False
x = base_model.output
# 層追加
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(y_train.shape[-1],
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation="softmax")(x)
model = tf.keras.Model(base_model.input, outputs)


#モデルコンパイル
model.compile(
  optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
  metrics=['accuracy'])

# base_model.summary()
# mymodel.summary()
model.summary()

#学習開始準備
steps_per_epoch  = math.ceil(x_train.shape[0] / BATCH_SIZE)
validation_steps = math.ceil(x_test.shape[0] / BATCH_SIZE)

print('*'*30)
print('BATCH_SIZE={}'.format(BATCH_SIZE))
print('x_train.shape[0]={}'.format(x_train.shape[0]))
print('steps_per_epoch={}'.format(steps_per_epoch))
print('x_test.shape[0]={}'.format(x_test.shape[0]))
print('validation_steps={}'.format(validation_steps))
print('*'*30)

#学習実行
model.fit(
    train_dataset,
    epochs=EPOCH_NUM, steps_per_epoch=steps_per_epoch,
    validation_data=test_dataset,
    validation_steps=validation_steps).history

#モデル保存
model.save("model_save")

######################
# tflite 作成
converter = tf.lite.TFLiteConverter.from_saved_model("model_save")
tflite_model = converter.convert()
dt_now_time = datetime.datetime.now()
str_now_time = "{}{:02}{:02}{:02}{:02}{:02}".format(dt_now_time.year,
                                                    dt_now_time.month,
                                                    dt_now_time.day,
                                                    dt_now_time.hour,
                                                    dt_now_time.minute,
                                                    dt_now_time.second)
with open("model_save/model_img_recog_pug_bull_FT_{}.tflite".format(str_now_time), 'wb') as o_:
    o_.write(tflite_model)
######################
