import tensorflow as tf


######################
# tflite 作成
converter = tf.lite.TFLiteConverter.from_saved_model("model_save")
tflite_model = converter.convert()
with open("model_img_recog_pug_bull_FT_byTFhub.tflite", 'wb') as o_:
    o_.write(tflite_model)
######################
