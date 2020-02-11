# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageGrab 

import sys
import time

def time_convert(sec):
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  return hours, mins, sec

#background subtraction
sub = cv2.createBackgroundSubtractorMOG2()

jumlah_pelanggaran = 0
pergi = 1
sekali_aja = 0
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
VIDEO_NAME = 'WIN_20191221_13_57_27_Pro (1).mp4'


#timer

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 2

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)

while(video.isOpened()):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    
    
    #define region of interest
    #roi = frame[200:500 , 100:1000]
        
    #cv2.rectangle(frame,(100,200),(1000,500),(0,255,0),5)
    roi = frame[400:900 , 500:1000]
        
    cv2.rectangle(frame,(500,400),(1000,900),(0,255,0),5)
    #hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    frame_expanded = np.expand_dims(roi, axis=0)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # converts image to gray
    fgmask = sub.apply(gray)  # uses the background subtraction
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
    closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("closing", closing) #@
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    #cv2.imshow("opening", opening) #@
    #dilation = cv2.dilate(opening, kernel)
    img_erosion = cv2.erode(opening, kernel, iterations=1) 
    #img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    #cv2.imshow("dilation", dilation) #@
    retvalbin, bins = cv2.threshold(img_erosion, 220, 255, cv2.THRESH_BINARY)  # removes the shadows

    #making background subtraction to countour
    contours, hierarchy = cv2.findContours(bins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    car_is_moving = False
    minarea = 1000
    maxarea = 50000
    for i in range(len(contours)):
      area = cv2.contourArea(contours[i])  # area of contour
      if minarea < area < maxarea:  # area threshold for contour
        car_is_moving = True
        break
    #print(car_is_moving)
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        roi,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        #np.squeeze(round(time.time()-start_time,0)),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)

    # All the results have been drawn on the frame, so it's time to display it.
    #print(classes).astype(np.int32)
    #print(num_detections)
    X = np.squeeze(scores)
    s_class = classes[scores > 0.5]
    end_time = time.time()
    str = "terjadi {0} pelanggaran".format(jumlah_pelanggaran)
    cv2.putText(frame, str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),3)
    if not s_class.any() :
        print("tidak ada mobil",s_class)
        pergi = 1
        sekali_aja = 0

    else:
        print('ada mobil',s_class)
        if car_is_moving == False:
          if pergi == 1:
              start_time = time.time()
              pergi = 0
          end_time = time.time()
          time_lapsed = end_time - start_time
          jam, menit, detik = time_convert(time_lapsed)
          a=("Time Lapsed = {0}:{1}:{2}".format(int(jam),int(menit),int(detik)))
          cv2.putText(frame, a, (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),3)

          if int(detik)>= 30 or int(menit)>0 or int(jam)>0:
              #im2 = ImageGrab.grab(bbox =(500, 400, 1000, 900)) 
              #im2.show() 
              print("ada orang parkir")
              if sekali_aja == 0:
                  jumlah_pelanggaran = jumlah_pelanggaran + 1
                  sekali_aja = 1
              print ("terjadi {0} pelanggaran".format(jumlah_pelanggaran))
         
    #print(X)
    #if(classes.all() ==1):
        #print('test')
    cv2.imshow('Object detector', frame)
    cv2.imshow('cuy',roi)
    cv2.imshow('background',bins)



    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break


      
# Importing Image and ImageGrab module from PIL package  
    
# creating an image object 
#im1 = Image.open(r"C:\Users\sadow984\Desktop\download2.JPG") 
    
# using the grab method 


# Clean up
video.release()
cv2.destroyAllWindows()
