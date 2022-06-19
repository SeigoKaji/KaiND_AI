import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

VISUALISE_CLASS_INDEX = 0

def grad_cam(model, img, layername, idx):

    # RGB-->BGR
    uint8_BGRimg = cv2.cvtColor((img*255).astype('uint8'), cv2.COLOR_RGB2BGR)
    img = np.expand_dims(img, axis=0)


    model_for_gradcam = tf.keras.models.Model([model.inputs], [model.get_layer(layername).output, model.output])

    # 勾配計算
    with tf.GradientTape() as tape:
        conv_outputs, predictions = model_for_gradcam(img)
        predict = predictions[:, VISUALISE_CLASS_INDEX]

    output = conv_outputs[0]
    grads = tape.gradient(predict, conv_outputs)[0]

    ### GradCamの論文通りの実装にするために、guidedはしない
    # gate_f = tf.cast(output > 0, 'float32')
    # gate_r = tf.cast(grads > 0, 'float32')
    # guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
    guided_grads = grads

    weights = tf.reduce_mean(guided_grads, axis=(0, 1))

    cam = np.ones(output.shape[0: 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam.numpy(), (224, 224))
    cam = np.maximum(cam, 0) # ReLU
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())


    cam_heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

    output_image = cv2.addWeighted(uint8_BGRimg, 0.5, cam_heatmap, 1, 0)


    # images save
    output_parent_dir = "./grad_cam_result"

    output_dir_path = output_parent_dir + "/input"
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    cv2.imwrite(output_dir_path + '/input_img_{:04}.png'.format(idx), uint8_BGRimg)

    output_dir_path = output_parent_dir + "/output_csv"
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    np.savetxt(output_dir_path + '/cam_{:04}.csv'.format(idx), cam, delimiter=',', fmt='%f')

    output_dir_path = output_parent_dir + "/output"
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    cv2.imwrite(output_dir_path + '/cam_{:04}.png'.format(idx), output_image)
