from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import detect_face
import random
from time import sleep
import time 
import shutil

def align_face(input_dir, sess, pnet, rnet, onet):
    sleep(random.random())

    output_dir = "Processed"
    image_size = 160
    margin = 32
    random_order = True
    gpu_memory_fraction = 0.25
    detect_multiple_faces = False

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    dataset = facenet.get_dataset(input_dir)

    print("Creating networks and loading parameters")
    
    min_size = 20 # minimum size of face
    threshold = [0.6, 0.7, 0.7] #three steps's threshold
    factor = 0.709 # scale factor

    nrof_images_total = 0
    nrof_successfully_aligned = 0
    if random_order:
        random.shuffle(dataset)
    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir) # create folder Name_ID in processed folder
            if random_order:
                random.shuffle(cls.image_paths)
        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename+'.png')
            print(image_path)
            if not os.path.exists(output_filename):
                try: 
                    img = misc.imread(image_path)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'. format(image_path, e)
                    print(errorMessage)
                else:
                    if img.ndim < 2:
                        print("Unable to align '%s'" % image_path)
                        #text_file.write("%s\n" % (output_filename))
                        continue
                    if img.ndim == 2:
                        img = facenet.to_rgb(img)
                    img = img[:, :, 0:3]
                    bounding_boxes, _ = detect_face.detect_face(img, min_size, pnet, rnet, onet, threshold, factor)

                    nrof_faces = bounding_boxes.shape[0]
                    if nrof_faces > 0:
                        det = bounding_boxes[:,0:4]
                        det_arr = []
                        img_size = np.asarray(img.shape)[0:2]
                        if nrof_faces > 1:
                            if detect_multiple_faces:
                                for i in range(nrof_faces):
                                    det_arr.append(np.squeeze(det[i]))
                            else:
                                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                                img_center = img_size / 2
                                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                                index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                                det_arr.append(det[index,:])
                        else:
                            det_arr.append(np.squeeze(det))
                            
                        for i, det in enumerate(det_arr):
                            det = np.squeeze(det)
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(det[0]-margin/2, 0)
                            bb[1] = np.maximum(det[1]-margin/2, 0)
                            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                            scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                            nrof_successfully_aligned += 1
                            filename_base, file_extension = os.path.splitext(output_filename)
                            if detect_multiple_faces:
                                output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                            else:
                                output_filename_n = "{}{}".format(filename_base, file_extension)
                            misc.imsave(output_filename_n, scaled)
                            #text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                    else:
                        print('Unable to align "%s"' % image_path)
                        #with open(miss_image_file, "a") as f:
                        #text_file.write('%s\n' % (output_filename))
                            #f.write('%s\n' % (output_filename))
    
    #shutil.rmtree(input_dir, ignore_errors=True)
    
    return nrof_images_total, nrof_successfully_aligned
    
