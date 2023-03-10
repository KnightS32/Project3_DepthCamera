## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2 as cv

# initialize tracker
tracker = cv.TrackerKCF_create()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

clipping_distance = 1 / device.first_depth_sensor().get_depth_scale()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)


# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)


# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# blank_image = np.zeros((640,1280,3), dtype=np.uint8)

for i in range(30):
    frames = pipeline.wait_for_frames()

color_frame = frames.get_color_frame()
color_image = np.asanyarray(color_frame.get_data())

# Uncomment the line below to select a different bounding box
bbox = cv.selectROI(color_image, False)

# Initialize tracker with first frame and bounding box
ok = tracker.init(color_image, bbox)

def getAvgDepth(bbox):
    avg = 0
    area = ((bbox[3] - bbox[1]) * (bbox[2] - bbox[0]))
    if area > 0:
        scale = 1 / area
        for i in range(bbox[0], bbox[2]):
            for k in range(bbox[1], bbox[3]):
                avg += depth_image[k, i] * scale
    print('avg depth: ', avg)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        v, h = depth_image.shape
        blank_image = np.zeros((v, 2 * h, 3), dtype=np.uint8)

        ok, bbox = tracker.update(color_image)
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv.rectangle(color_image, p1, p2, (255, 0, 0), 2, 1)
        else:

 
            #color_copy = color_image.copy()
            #gray_img = cv.cvtColor(color_copy, cv.COLOR_BGR2GRAY)

            #ret, thresh = cv.threshold(gray, 127, 255, 0)

            #contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            #cnt = contours[0]
            
            #x,y,w,h = cv.boundingRect(cnt)

            #color_copy = cv.rectangle(color_copy,(x,y),(x+w, y+h),(0,255,0),2) 
        

            # if tracking failed, find new object to track
			# go through different clipping distances from closest to farthest
            for i in range(1, 5):
                clipping_distance = i * 0.5 / depth_scale
                grey_color = 153
                depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
                # remove distant background from area
                color_image_copy = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
                gray_img = cv.cvtColor(color_image_copy, cv.COLOR_BGR2GRAY)
                ret, thresh = cv.threshold(gray_img, 127, 255, 0)
                contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                cnt = contours[0]
                #x,y,w,h = cv.boundingRect(cnt)
                ok = tracker.init(color_image, cv.boundingRect(cnt))
                #color_copy = cv.rectangle(color_copy,(x,y),(x+w, y+h),(0,255,0),2)
			# pass this to find contour thing
			# if the contour is significant
			# use bounding box of contour as new bounding box
			
			# We will be removing the background of objects more than
            #  clipping_distance_in_meters meters away


            # Tracking failure
            cv.putText(color_image, "Tracking failure detected", (100, 80), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # getAvgDepth(bbox)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape
        
        blank_image = cv.rectangle(blank_image, (h-2,v//2-5), (h+2,v//2+5), (0, 0, 255), -1)

        if(abs((bbox[3] - bbox[1]) * (bbox[2] - bbox[0]))>0):
            # center point of bounding box
            xC = bbox[3] - (bbox[3]//2 - bbox[1])
            yC = bbox[2] - (bbox[2]//2 - bbox[0])
            # get depth, scaled off 2 meters
            depth = depth_image[xC, yC] / (2 / depth_scale)
            # convert to y coordinate
            depth = int (blank_image.shape[0]//2 - blank_image.shape[0]//2 * depth)
            # get scale of blank image width to color image width
            wscale = (blank_image.shape[1] / color_image.shape[1])
            # make points for blank image
            p1 = ((int) (bbox[0] * wscale), depth)
            p2 = ((int) ((bbox[0] + bbox[2]) * wscale), depth)
            # do a thing
            cv.rectangle(blank_image, p1, p2, (0, 255, 0), -1)

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                            interpolation=cv.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
            images = np.vstack((images, blank_image))
        else:
            images = np.hstack((color_image, depth_colormap))
            images = np.vstack((images, blank_image))

        # Show images
        cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)
        cv.imshow('RealSense', images)

        # exit program by pressing key 'esc'
        if cv.waitKey(1) == 27:
            break

finally:
    # Stop streaming
    pipeline.stop()
