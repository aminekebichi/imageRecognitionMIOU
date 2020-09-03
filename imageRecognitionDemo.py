# image detection application
# 09/02/2020

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
import zipfile
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from os import listdir
from os.path import isfile, join
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from os import listdir
from os.path import isfile, join

os.chdir("C:\\Users\\amine\\EDIN\\Public\\models\\research") 
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v. 1.4.0')

%matplotlib inline

#Object detection imports
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

MODEL_NAME = 'faster_rcnn_resnet101_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = 'C:/Users/amine/EDIN/Public/models/research/' + MODEL_NAME + '\\frozen_inference_graph.pb'
PATH_TO_SAVED_MODEL = MODEL_NAME + '/saved_model/saved_model.pb'
LOGDIR= MODEL_NAME + '/logs'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

#Download the model
import tarfile
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inferene_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())
    if 'saved_model.pb' in file_name:
        tar_file.extract(file, os.getcwd())

#Load a (frozen) TF model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

#Loading label map
os.chdir("C:\\Users\\amine\\EDIN\\Public\\models\\research\\object_detection")
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
    (im_height, im_width, 3)).astype(np.uint8)

#Detection
PATH_TO_TEST_IMAGES_DIR = 'C:\\Users\\amine\\Downloads\\val2017'
TEST_IMAGE_PATHS = [f for f in listdir(PATH_TO_TEST_IMAGES_DIR) if isfile(join(PATH_TO_TEST_IMAGES_DIR, f))]
IMAGE_SIZE = (12, 8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})      
        
      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

#for image_path in TEST_IMAGE_PATHS:
if (len(TEST_IMAGE_PATHS) > 1):
  image_path = TEST_IMAGE_PATHS[0]
  image = Image.open(join(PATH_TO_TEST_IMAGES_DIR, image_path))
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
  plt.figure(figsize=IMAGE_SIZE)
  plt.imshow(image_np)
  #print (output_dict['detection_boxes'])
  #print (output_dict.get('detection_masks'))

import time
import json

total_time = 0.0;
grbboxes_filenames=[]
img_bbox_ids=[]
filename_img_bbox_ids=[]

def gr_bbox():
    json_filename="C:\\Users\\amine\\Desktop\\EDIN\\instances_val2017.json"
    array=[]
    try:
        with open(json_filename) as data_file:
            data=json.load(data_file)
            img_height=0
            img_width=0
            for each_axis in data['images']:
                img_height = each_axis['height']
                img_width = each_axis['width']
                img_filename = each_axis['file_name']
                grbboxes_filenames.append(img_filename)
                image_id = each_axis['id']
                filename_img_bbox_ids.append(image_id)
                
            if len(grbboxes_filenames) != len(filename_img_bbox_ids):
                print ("these 2 should be the same length")
            
            for each_axis in data['annotations']:
                image_id =  each_axis['image_id']
                img_bbox_ids.append(image_id)
                X = each_axis['bbox']
                if image_id == 139:
                    #print ("img id", image_id)
                    print ("bbox", X)
                tmp_bb=[0, 0, 0, 0]
                tmp_bb[0]=X[2]/img_height
                tmp_bb[1]=X[3]/img_width
                tmp_bb[2]=X[0]/img_height
                tmp_bb[3]=X[1]/img_width
                X=tmp_bb
                if image_id == 139:
                    print ("bbox normalized", X)
                array.append(X)
                
            if len(img_bbox_ids) != len(array):
                print ("These 2 should also have same length")
                
    except:
        print("Unexpected error", sys.exc_info()[0])
        raise
        
    gr_bboxes = list()
    #max_coord=np.max(array)
    for bbox in array:
        #print (bbox)
        #bbox = [x/max_coord for x in bbox[:4]]
        #bbox = [x for x in bbox[:4]]
        #bbox = [x/256 for x in bbox[:4]]
        gr_bboxes.append(bbox)
        
    return gr_bboxes
        
gr_bboxes = gr_bbox()