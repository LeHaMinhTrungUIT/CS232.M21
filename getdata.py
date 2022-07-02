# lấy dữ liệu để train từ cam

# từ khóa __future__ để cập nhật các tính năng mới của các hàm được gọi
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import argparse
import facenet
import os
import sys
import math
import pickle
import detect_face
import numpy as np
import cv2
import collections
import shutil
from sklearn.svm import SVC
from scipy import misc
import pandas as pd


MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.7
INPUT_IMAGE_SIZE = 160
FACENET_MODEL_PATH = './model/20180402-114759.pb'
tf.Graph().as_default()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.Session(config=tf.ConfigProto(
    gpu_options=gpu_options, log_device_placement=False))

name = input('Nhập tên:')

if not os.path.isfile("labels.csv"):
    with open("labels.csv", "a") as f:
        f.write('0,')
        f.write(name)
        f.write('\n')
else:
    df_labels = pd.read_csv("labels.csv")
    my_ids = df_labels.shape[0]
    my_ids += 1
    my_ids = [my_ids]
    sav = pd.concat([pd.DataFrame(my_ids), pd.Series(name)], axis=1)
    sav.to_csv("labels.csv", mode='a', header=None, index=False)

folder = "Dataset/" + str(name) + "/"
if not os.path.exists(folder):
    os.makedirs(folder)

# Load the model
print('Loading feature extraction model')
facenet.load_model(FACENET_MODEL_PATH)

# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
pnet, rnet, onet = detect_face.create_mtcnn(sess, "./align")

# , model, class_names, sess, images_placeholder, embeddings, phase_train_placeholder, pnet, rnet, onet


def Extract_feature():
    # Time start
    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.7
    INPUT_IMAGE_SIZE = 160

    if not os.path.exists("Dataset"):
        os.mkdir("Dataset")

    cap = cv2.VideoCapture(0)
    cnt = 0
    while (True):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)

        bounding_boxes, _ = detect_face.detect_face(
            frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
        faces_found = bounding_boxes.shape[0]
        print("faces_found: ", faces_found)

        if bounding_boxes != []:
            flag = 1
            # for person in bounding_boxes:
            det = bounding_boxes[:, 0:4]
            bb = np.zeros((faces_found, 4), dtype=np.int32)
            for i in range(faces_found):
                bb[i][0] = det[i][0]
                bb[i][1] = det[i][1]
                bb[i][2] = det[i][2]
                bb[i][3] = det[i][3]
                cv2.rectangle(frame, (bb[i][0], bb[i][1]),
                              (bb[i][2], bb[i][3]), (0, 255, 0), 2)

                if (bb[i][3]-bb[i][1])/frame.shape[0] > 0.0:

                    cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]

                    scaled_out = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                            interpolation=cv2.INTER_CUBIC)

                    scaled = facenet.prewhiten(scaled_out)
                    scaled_reshape = scaled.reshape(-1,
                                                    INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                    feed_dict = {images_placeholder: scaled_reshape,
                                 phase_train_placeholder: False}
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)

                    print("emb_array.shape: ", emb_array.shape)
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)
                    # lưu lại ảnh vào folder data
                    cv2.imwrite(folder + str(cnt) + '.jpg', scaled_out)
                    emb_array = np.append(emb_array, name)
                    my_features = np.array(emb_array)
                    my_features = my_features.reshape(-1, my_features.shape[0])
                    df = pd.DataFrame(my_features)
                    df.to_csv("features.csv", mode='a',
                              header=None, index=False)
                    cnt += 1

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cnt > 100:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    Extract_feature()
