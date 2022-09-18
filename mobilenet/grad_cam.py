import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def grad_cam(model, img, layername, visualise_class):

    # RGB-->BGR
    uint8_BGRimg = cv2.cvtColor((img*255).astype('uint8'), cv2.COLOR_RGB2BGR)
    img = np.expand_dims(img, axis=0)


    model_for_gradcam = tf.keras.models.Model([model.inputs], [model.get_layer(layername).output, model.output])

    # 勾配計算
    with tf.GradientTape() as tape:
        conv_outputs, predictions = model_for_gradcam(img)
        predict = predictions[:, visualise_class]

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

    return uint8_BGRimg, heatmap, output_image

def save_grad_cam_outputs(uint8_BGRimg, heatmap, output_image, idx, classname):
    # images save
    output_parent_dir = "./grad_cam_result"

    output_dir_path = os.path.join(output_parent_dir, "input", classname)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    cv2.imwrite(os.path.join(output_dir_path, 'input_img_{:04}.png'.format(idx)), uint8_BGRimg)

    # output_dir_path = os.path.join(output_parent_dir, "output_csv", classname)
    # if not os.path.exists(output_dir_path):
    #     os.makedirs(output_dir_path)
    # np.savetxt(os.path.join(output_dir_path, 'cam_{:04}.csv'.format(idx)), heatmap, delimiter=',', fmt='%f')

    output_dir_path = os.path.join(output_parent_dir, "output", classname)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    cv2.imwrite(os.path.join(output_dir_path, 'cam_{:04}.png'.format(idx)), output_image)

def img_packing_to_largeimg(largeimg, packing_img, h_s, w_s):
    packing_shape_h = packing_img.shape[0]
    packing_shape_w = packing_img.shape[1]
    largeimg[h_s:h_s+packing_shape_h, w_s:w_s+packing_shape_w, 0] = packing_img[:, :, 0]
    largeimg[h_s:h_s+packing_shape_h, w_s:w_s+packing_shape_w, 1] = packing_img[:, :, 1]
    largeimg[h_s:h_s+packing_shape_h, w_s:w_s+packing_shape_w, 2] = packing_img[:, :, 2]

    return largeimg

def class_num(output_img, puttext, h_s, w_s):
    cv2.putText(output_img, text=puttext)
    return output_img

def save_grad_cam_outputs_to_oneimg(uint8_BGRimg, heatmaps, idx, class_num, predict, y_test, classidx2classname):
    # images save
    output_parent_dir = "./grad_cam_result"

    out_shape = uint8_BGRimg.shape

    out_img = np.zeros((out_shape[0]*2, out_shape[1]*class_num, 3))

    out_img = img_packing_to_largeimg(out_img, uint8_BGRimg, 0, 0)
    for i in range(class_num):
        out_img = img_packing_to_largeimg(out_img, heatmaps[i], out_shape[1], out_shape[0]*i)

    # print('y_test:', y_test.shape, type(y_test))
    # print(y_test)
    PUTTEXT_HEGIHT_PARTITION = 6
    # 正解値クラスが何かを表記
    str_gt_class = classidx2classname[np.argmax(y_test)]
    str_gt_class_info = "gt class: {}".format(str_gt_class)
    # print('str_gt_class:', str_gt_class)
    cv2.putText(out_img, str_gt_class_info, org=(out_shape[0]+int(out_shape[0]/10), int(out_shape[1]/PUTTEXT_HEGIHT_PARTITION)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)

    # 各クラスのスコアを表記
    for i in range(class_num):
        classname = classidx2classname[i]
        score = predict[i]
        # print('classname:', classname)
        # print('score:', score)
        str_pred_class_info = "{}:{:.6f}".format(classname, score)
        cv2.putText(out_img, str_pred_class_info, org=(out_shape[0]+int(out_shape[0]/10), int(out_shape[1]/PUTTEXT_HEGIHT_PARTITION)*(i+2)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA)

    # 各ヒートマップに推定クラス名を表記
    for i in range(class_num):
        classname = classidx2classname[i]
        # print('classname:', classname)
        str_classname_info = "{}".format(classname, score)
        cv2.putText(out_img, str_classname_info, org=(out_shape[0]*(i)+int(out_shape[0]/10), out_shape[1]+int(out_shape[1]/PUTTEXT_HEGIHT_PARTITION)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,0,0), thickness=2, lineType=cv2.LINE_AA)


    output_dir_path = os.path.join(output_parent_dir, "results")
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    cv2.imwrite(os.path.join(output_dir_path, 'result_{:04}.png'.format(idx)), out_img)