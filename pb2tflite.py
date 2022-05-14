import os
import tensorflow as tf

output_dir_path = "../trained_models/model_img_recog_ramen_pan_bn"
# if not os.path.exists(output_dir_path):
#     os.makedirs("../trained_models/model_img_recog_ramen_pan_bn")
# model.save("../trained_models/model_img_recog_ramen_pan_bn")
# model.save("../trained_models/model_img_recog_ramen_pan_bn.h5")

input_model = output_dir_path + "/saved_model.pb"
# input_name = "model_input"
# output_node_name = "mobilenetv2_1.00_224/Logits/Softmax"

output_model = "../trained_models/model_img_recog_ramen_pan_bn.tflite"

#to tensorflow lite
# converter = tf.lite.TFLiteConverter.from_frozen_graph(input_model,[input_name],[output_node_name])
converter = tf.lite.TFLiteConverter.from_saved_model(output_dir_path)
tflite_model = converter.convert()
with open(output_model, 'wb') as o_:
    o_.write(tflite_model)