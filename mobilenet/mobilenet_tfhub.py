#参考URL
#https://www.tensorflow.org/hub?hl=ja
#https://tfhub.dev/s?module-type=image-feature-vector&tf-version=tf2
import tensorflow as tf
import tensorflow_hub as hub

#設定
BATCH_SIZE = 32
EPOCH_NUM = 20
do_fine_tuning = False
# do_fine_tuning = True
IMAGE_SIZE = (224, 224)

#事前学習済みモデル取得
handle_base = "mobilenet_v2_100_224"
MODULE_HANDLE ="https://tfhub.dev/google/imagenet/{}/feature_vector/5".format(handle_base)

#学習データ取得
data_dir = "train_data"

#画像前処理情報
# datagen_kwargs = dict(rescale=1./255, validation_split=.20)
test_datagen_kwargs  = dict(rescale=1./255)
train_datagen_kwargs = dict(rescale=1./255, horizontal_flip=True, vertical_flip=True,
                        rotation_range=45, width_shift_range=0.2, height_shift_range=0.2,
                        brightness_range=[0.5, 1.0], zoom_range=0.3)
dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                   interpolation="bilinear")

#テストデータ用意
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **test_datagen_kwargs)
test_generator = test_datagen.flow_from_directory(
    data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

#訓練データ用意
# train_datagen = test_datagen
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **train_datagen_kwargs)
train_generator = train_datagen.flow_from_directory(
    data_dir, subset="training", shuffle=True, **dataflow_kwargs)


#識別器部分追加
model = tf.keras.Sequential([
    # tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE+(3,)),
    hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(train_generator.num_classes,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation="softmax")
])
model.build((None,)+IMAGE_SIZE+(3,))
model.summary()

#モデルコンパイル
model.compile(
  optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
  metrics=['accuracy'])

#学習開始準備
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = test_generator.samples // test_generator.batch_size

#学習実行
model.fit(
    train_generator,
    epochs=EPOCH_NUM, steps_per_epoch=steps_per_epoch,
    validation_data=test_generator,
    validation_steps=validation_steps).history

#モデル保存
tf.saved_model.save(model, "model_save")

######################
# tflite 作成
converter = tf.lite.TFLiteConverter.from_saved_model("model_save")
tflite_model = converter.convert()
with open("model_img_recog_pug_bull_FT_byTFhub.tflite", 'wb') as o_:
    o_.write(tflite_model)
######################
