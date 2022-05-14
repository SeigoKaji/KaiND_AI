import os
import tensorflow as tf
from load import Get_Training_and_Test_Image

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
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
print(train_dataset)
# shuffle
# batch
train_dataset = train_dataset.shuffle(x_train.shape[0]).batch(set_batch_size)
test_dataset = test_dataset.batch(set_batch_size)

######################NN構築(CNN)##############################
input_shape = (224, 224, 3)

# モデル構築
inputs = tf.keras.Input(shape=input_shape, name='input')

x = tf.keras.layers.Conv2D(64,  kernel_size=(7, 7), activation=None, use_bias=False, padding='same', strides = 2)(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

x1 = tf.keras.layers.Conv2D(64,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x)
x1 = tf.keras.layers.BatchNormalization()(x1)
x1 = tf.keras.layers.ReLU()(x1)
x2 = tf.keras.layers.Conv2D(64,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x1)
x2 = tf.keras.layers.BatchNormalization()(x2)

x2_added = tf.keras.layers.add([x2, x])
x2_added = tf.keras.layers.ReLU()(x2_added)

x3 = tf.keras.layers.Conv2D(64,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x2_added)
x3 = tf.keras.layers.BatchNormalization()(x3)
x3 = tf.keras.layers.ReLU()(x3)
x4 = tf.keras.layers.Conv2D(64,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x3)
x4 = tf.keras.layers.BatchNormalization()(x4)


x4_added = tf.keras.layers.add([x4, x2_added])
x4_added = tf.keras.layers.ReLU()(x4_added)

x5 = tf.keras.layers.Conv2D(64,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x4_added)
x5 = tf.keras.layers.BatchNormalization()(x5)
x5 = tf.keras.layers.ReLU()(x5)
x6 = tf.keras.layers.Conv2D(64,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x5)
x6 = tf.keras.layers.BatchNormalization()(x6)

x6_added = tf.keras.layers.add([x6, x4_added])
x6_added = tf.keras.layers.ReLU()(x6_added)

x7 = tf.keras.layers.Conv2D(128,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same', strides = 2)(x6_added)
x7 = tf.keras.layers.BatchNormalization()(x7)
x7 = tf.keras.layers.ReLU()(x7)
x8 = tf.keras.layers.Conv2D(128,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x7)
x8 = tf.keras.layers.BatchNormalization()(x8)

x6_added_2ch = tf.keras.layers.Conv2D(128,  kernel_size=(1, 1), activation=None, use_bias=False, padding='same', strides = 2)(x6)
x6_added_2ch = tf.keras.layers.BatchNormalization()(x6_added_2ch)
x6_added_2ch = tf.keras.layers.ReLU()(x6_added_2ch)
x8_added = tf.keras.layers.add([x8, x6_added_2ch])
x8_added = tf.keras.layers.ReLU()(x8_added)

x9 = tf.keras.layers.Conv2D(128,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x8_added)
x9 = tf.keras.layers.BatchNormalization()(x9)
x9 = tf.keras.layers.ReLU()(x9)
x10 = tf.keras.layers.Conv2D(128,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x9)
x10 = tf.keras.layers.BatchNormalization()(x10)

x10_added = tf.keras.layers.add([x10, x8_added])
x10_added = tf.keras.layers.ReLU()(x10_added)

x11 = tf.keras.layers.Conv2D(128,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x10_added)
x11 = tf.keras.layers.BatchNormalization()(x11)
x11 = tf.keras.layers.ReLU()(x11)
x12 = tf.keras.layers.Conv2D(128,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x11)
x12 = tf.keras.layers.BatchNormalization()(x12)

x12_added = tf.keras.layers.add([x12, x10_added])
x12_added = tf.keras.layers.ReLU()(x12_added)

x13 = tf.keras.layers.Conv2D(128,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x12_added)
x13 = tf.keras.layers.BatchNormalization()(x13)
x13 = tf.keras.layers.ReLU()(x13)
x14 = tf.keras.layers.Conv2D(128,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x13)
x14 = tf.keras.layers.BatchNormalization()(x14)

x14_added = tf.keras.layers.add([x14, x12_added])
x14_added = tf.keras.layers.ReLU()(x14_added)

x15 = tf.keras.layers.Conv2D(256,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same', strides = 2)(x14_added)
x15 = tf.keras.layers.BatchNormalization()(x15)
x15 = tf.keras.layers.ReLU()(x15)
x16 = tf.keras.layers.Conv2D(256,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x15)
x16 = tf.keras.layers.BatchNormalization()(x16)

x14_added_2ch = tf.keras.layers.Conv2D(256,  kernel_size=(1, 1), activation=None, use_bias=False, padding='same', strides = 2)(x14_added)
x14_added_2ch = tf.keras.layers.BatchNormalization()(x14_added_2ch)
x14_added_2ch = tf.keras.layers.ReLU()(x14_added_2ch)
x16_added = tf.keras.layers.add([x16, x14_added_2ch])
x16_added = tf.keras.layers.ReLU()(x16_added)

x17 = tf.keras.layers.Conv2D(256,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x16_added)
x17 = tf.keras.layers.BatchNormalization()(x17)
x17 = tf.keras.layers.ReLU()(x17)
x18 = tf.keras.layers.Conv2D(256,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x17)
x18 = tf.keras.layers.BatchNormalization()(x18)

x18_added = tf.keras.layers.add([x18, x16_added])
x18_added = tf.keras.layers.ReLU()(x18_added)

x19 = tf.keras.layers.Conv2D(256,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x18_added)
x19 = tf.keras.layers.BatchNormalization()(x19)
x19 = tf.keras.layers.ReLU()(x19)
x20 = tf.keras.layers.Conv2D(256,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x19)
x20 = tf.keras.layers.BatchNormalization()(x20)

x20_added = tf.keras.layers.add([x20, x18_added])
x20_added = tf.keras.layers.ReLU()(x20_added)

x21 = tf.keras.layers.Conv2D(256,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x20_added)
x21 = tf.keras.layers.BatchNormalization()(x21)
x21 = tf.keras.layers.ReLU()(x21)
x22 = tf.keras.layers.Conv2D(256,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x21)
x22 = tf.keras.layers.BatchNormalization()(x22)

x22_added = tf.keras.layers.add([x22, x20_added])
x22_added = tf.keras.layers.ReLU()(x22_added)

x23 = tf.keras.layers.Conv2D(256,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x22_added)
x23 = tf.keras.layers.BatchNormalization()(x23)
x23 = tf.keras.layers.ReLU()(x23)
x24 = tf.keras.layers.Conv2D(256,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x23)
x24 = tf.keras.layers.BatchNormalization()(x24)

x24_added = tf.keras.layers.add([x24, x22_added])
x24_added = tf.keras.layers.ReLU()(x24_added)

x25 = tf.keras.layers.Conv2D(256,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x24_added)
x25 = tf.keras.layers.BatchNormalization()(x25)
x25 = tf.keras.layers.ReLU()(x25)
x26 = tf.keras.layers.Conv2D(256,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x25)
x26 = tf.keras.layers.BatchNormalization()(x26)

x26_added = tf.keras.layers.add([x26, x24_added])
x26_added = tf.keras.layers.ReLU()(x26_added)

x27 = tf.keras.layers.Conv2D(512,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same', strides = 2)(x26_added)
x27 = tf.keras.layers.BatchNormalization()(x27)
x27 = tf.keras.layers.ReLU()(x27)
x28 = tf.keras.layers.Conv2D(512,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x27)
x28 = tf.keras.layers.BatchNormalization()(x28)

x26_added_2ch = tf.keras.layers.Conv2D(512,  kernel_size=(1, 1), activation=None, use_bias=False, padding='same', strides = 2)(x26_added)
x26_added_2ch = tf.keras.layers.BatchNormalization()(x26_added_2ch)
x26_added_2ch = tf.keras.layers.ReLU()(x26_added_2ch)
x28_added = tf.keras.layers.add([x28, x26_added_2ch])
x28_added = tf.keras.layers.ReLU()(x28_added)

x29 = tf.keras.layers.Conv2D(512,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x28_added)
x29 = tf.keras.layers.BatchNormalization()(x29)
x29 = tf.keras.layers.ReLU()(x29)
x30 = tf.keras.layers.Conv2D(512,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x29)
x30 = tf.keras.layers.BatchNormalization()(x30)

x30_added = tf.keras.layers.add([x30, x28_added])
x30_added = tf.keras.layers.ReLU()(x30_added)

x31 = tf.keras.layers.Conv2D(512,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x30_added)
x31 = tf.keras.layers.BatchNormalization()(x31)
x31 = tf.keras.layers.ReLU()(x31)
x32 = tf.keras.layers.Conv2D(512,  kernel_size=(3, 3), activation=None, use_bias=False, padding='same')(x31)
x32 = tf.keras.layers.BatchNormalization()(x32)

x32_added = tf.keras.layers.add([x32, x30_added])
x32_added = tf.keras.layers.ReLU()(x32_added)

x32_added = tf.keras.layers.AveragePooling2D((7, 7))(x32_added)

xout = tf.keras.layers.Flatten()(x32_added)

outputs = tf.keras.layers.Dense(len(dataset_cate_name), activation='softmax')(xout)


#層をセット(作ったパーツの組み合わせ終わったものをここでセッティング)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name="img_recog_model")

model.summary()

######################NN構築##############################

#モデルのコンパイル
sgd = tf.keras.optimizers.SGD(lr=0.005, decay=0.0001, momentum=0.9, nesterov=True)
model.compile(
#SparseCategoricalCrossentropy
loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
optimizer=sgd,
metrics=["accuracy"],
)

#学習
model.fit(x_train, y_train, epochs=set_epoch_num, batch_size=set_batch_size)

#評価
score = model.evaluate(x_test, y_test, verbose=2)
print('損失:', score[0])
print('正答率:', score[1])

#モデルの保存
model_name = "model_img_recog_pug_bull_bn"
output_dir_path = "../trained_models/" + model_name
if not os.path.exists(output_dir_path):
    os.makedirs(output_dir_path)
model.save(output_dir_path)
model.save(output_dir_path + ".h5")

input_model = output_dir_path + "/saved_model.pb"

output_model = "../trained_models/model_img_recog_pug_bull_bn.tflite"

#to tensorflow lite
converter = tf.lite.TFLiteConverter.from_saved_model(output_dir_path)
tflite_model = converter.convert()
with open(output_model, 'wb') as o_:
    o_.write(tflite_model)