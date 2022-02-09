#!/usr/bin/env python

'''
Object Detection Using CNN
(c) 2021 Paolo Rommel Sanchez
License BSD

Transmits coordinates using ROS
Code for multiple targets

Reference:
* DetectNet Methods: https://forums.developer.nvidia.com/t/detectnet-methods/184463/6
'''

from __future__ import print_function

import cv2              
import numpy as np                                                                                                                                                                                                                                                                         
import sys              
import time

# jetson inferencing
import jetson.inference
import jetson.utils

# ros libraries
import rospy
from std_msgs.msg import Int16MultiArray, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# camera parameters
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

class ImageCapture():

    def __init__(self):
        
        self.cap = jetson.utils.gstCamera(CAPTURE_WIDTH,CAPTURE_HEIGHT,"/dev/video0")
        self.net = jetson.inference.detectNet(argv=['--model=/home/vision-module/jetson-inference/python/training/detection/ssd/models/plants/ssd-mobilenet.onnx', \
            '--labels=/home/vision-module/jetson-inference/python/training/detection/ssd/models/plants/labels.txt',
            '--input-blob=input_0',
            '--output-cvg=scores',
            '--output-bbox=boxes',
            '--threshold=0.50'])

        self.bridge = CvBridge()
        self.pub_vision_weed_coordinates = rospy.Publisher('/vision/weed_coordinates_px', Int16MultiArray, queue_size=1)
        self.pub_vision_crop_coordinates = rospy.Publisher('/vision/crop_coordinates_px', Int16MultiArray, queue_size=1)
        self.pub_vision_images = rospy.Publisher('/vision/images', Image)
        self.pub_vision_time_delay = rospy.Publisher('/vision/time_delay_s', Float32)

        # used to record the time when we processed last frame
        self.prev_frame_time = time.time()
        
        # used to record the time at which we processed current frame
        self.new_frame_time = 0.


    def get_coordinate(self): # loop each captured frame
    
        im,width,height = self.cap.CaptureRGBA(zeroCopy=1) # get camera value, zeroCopy to not create new instance
        detections = self.net.Detect(im,width,height)	# perform detections
        im = jetson.utils.cudaToNumpy(im,width,height,4) # conversion from CUDA to numpy to support cv2 format
        
        jetson.utils.cudaDeviceSynchronize() # wait for GPU to finish processing before CPU-based execution

        weed_coordinate = []
        crop_coordinate = []
        if len(detections) > 0:
            i=0
            
            # weed
            weed_x_coordinates=[]
            weed_y_coordinates=[]
            
            # crop
            crop_x_coordinates=[]
            crop_y_coordinates=[]


            for detection in detections:      # loop each detected object
                
                # object counter
                i+=1

                if detection.ClassID==1:
                    crop_x_coordinates.append(int(detection.Center[0]))
                    crop_y_coordinates.append(int(detection.Center[1]))
                elif detection.ClassID==2:
                # object coordinates
                    weed_x_coordinates.append(int(detection.Center[0]))
                    weed_y_coordinates.append(int(detection.Center[1]))
            
            if len(crop_x_coordinates) !=0:
                crop_coordinate = crop_x_coordinates
            else:
                crop_coordinate = [-1]

            if len(weed_x_coordinates) !=0:
                weed_coordinate = weed_x_coordinates
            else:
                weed_coordinate = [-1]
        
        else:       # returns -1 if nothing is detected
            crop_coordinate = [-1]
            weed_coordinate = [-1]

        # effective transmission rate:
        self.new_frame_time = time.time()
        time_processing = self.new_frame_time-self.prev_frame_time
        fps = round(1./(time_processing),2)
        self.prev_frame_time = self.new_frame_time

        # Display FPS counter
        # can be skipped as cpu-based image processing causes significant inferencing delay
        im = cv2.cvtColor(im.astype(np.uint8),cv2.COLOR_RGBA2BGR) # convert to numpy format
        cv2.putText(im, "Effective Camera Speed | {:.0f} FPS".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(im, "Inference Speed | {:.0f} FPS".format(self.net.GetNetworkFPS()), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 0), 1, cv2.LINE_AA)

        # Display the image
        cv2.imshow("Detections",im)              # create a window to display the 'marked' frame

        # Add artificial delay
        #time.sleep(0.1)

        return crop_coordinate, weed_coordinate, im, time_processing

    def run(self):
        
        #rate = rospy.Rate(MAX_CAPTURE_FPS) # limit to max fps of camera, causes error due to assynchronous fetching of capturing and fetching
        while (not rospy.is_shutdown()):

            try:
                time_capture = time.time()
                crop_coordinates, weed_coordinates, im, time_processing = self.get_coordinate()

                # publish coordinates
                crop_coordinates = Int16MultiArray(data=crop_coordinates)
                weed_coordinates = Int16MultiArray(data=weed_coordinates)
                self.pub_vision_crop_coordinates.publish(crop_coordinates)
                self.pub_vision_weed_coordinates.publish(weed_coordinates)

                # publish images
                # Warning: Transmitting large messages such as images in the network slows the overall effective transmission rate
                #self.pub_vision_images.publish(self.bridge.cv2_to_imgmsg(im, "bgr8"))

                # publish delay time
                time_delay = time.time() - time_capture
                self.pub_vision_time_delay.publish(time_delay)
                
                # Warning: Displaying multiArray messages using rospy loginfo causes slow transmission rate
                #rospy.loginfo(weed_coordinates)
                #rospy.loginfo(crop_coordinates)
                #rospy.loginfo(1./time_delay)

            except CvBridgeError as e:
                print(e)

            finally:
                #rate.sleep()

                if cv2.waitKey(1) & 0xFF == ord('q'):           # press 'q' to exit; otherwise, the script will continue refreshing the frame
                    break


        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    
    try:
        rospy.init_node('vision_module')
        capture = ImageCapture()
        capture.run()

    except rospy.ROSInterruptException:
        pass

    finally:
        # clear the system RAM
        sys.modules[__name__].__dict__.clear()
        print("Exited node.")
        exit
