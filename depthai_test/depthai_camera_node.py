#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import depthai as dai
import numpy as np
import time
import threading



class DepthAICameraNode(Node):
    def __init__(self):
        super().__init__('depthai_camera_node')
        self.declare_parameter('model_path', '/')
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        model_path = self.get_parameter('model_path').get_parameter_value().string_value 
        self.publisher_ = self.create_publisher(Image, 'detection/image', qos_profile)
        self.bridge = CvBridge()
        # Define sources and outputs
        fullFrameTracking = False
        self.labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        self.pipeline = dai.Pipeline()
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        detectionNetwork = self.pipeline.create(dai.node.MobileNetDetectionNetwork)
        objectTracker = self.pipeline.create(dai.node.ObjectTracker)

        xlinkOut = self.pipeline.create(dai.node.XLinkOut)
        trackerOut = self.pipeline.create(dai.node.XLinkOut)

        xlinkOut.setStreamName("preview")
        trackerOut.setStreamName("tracklets")

        # Properties
        camRgb.setPreviewSize(300, 300)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camRgb.setFps(40)

        # testing MobileNet DetectionNetwork
        detectionNetwork.setBlobPath(model_path)
        detectionNetwork.setConfidenceThreshold(0.5)
        detectionNetwork.input.setBlocking(False)

        objectTracker.setDetectionLabelsToTrack([15])  # track only person
        # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
        objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
        # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
        objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        # Linking
        camRgb.preview.link(detectionNetwork.input)
        objectTracker.passthroughTrackerFrame.link(xlinkOut.input)

        if fullFrameTracking:
            camRgb.video.link(objectTracker.inputTrackerFrame)
        else:
            detectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

        detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
        detectionNetwork.out.link(objectTracker.inputDetections)
        objectTracker.out.link(trackerOut.input)
        # self.device = dai.Device(self.pipeline)
        # self.preview = self.device.getOutputQueue("preview", 4, False)
        # self.tracklets = self.device.getOutputQueue("tracklets", 4, False)

        # self.startTime = time.monotonic()
        # self.counter = 0
        # self.fps = 0
        # self.frame = None
        # Create a thread for the timer callback
        self.camera_thread = threading.Thread(target=self.depth_camera_thread)
        self.camera_thread.daemon = True  # Set as a daemon thread
        self.camera_thread.start()

    def print_in_yellow(self, msg):
        yellow_start = "\033[33m"  # Yellow color.
        color_end = "\033[0m"
        self.get_logger().info(f"{yellow_start}{msg}{color_end}")

    def print_in_green(self, msg):
        green_start = "\033[92m"
        color_end = "\033[0m"
        self.get_logger().info(f"{green_start}{msg}{color_end}")

    def depth_camera_thread(self):
        with dai.Device(self.pipeline) as device:
            preview = device.getOutputQueue("preview", 4, False)
            tracklets = device.getOutputQueue("tracklets", 4, False)

            startTime = time.monotonic()
            counter = 0
            fps = 0
            frame = None
            while rclpy.ok():
                imgFrame = preview.get()
                track = tracklets.get()
                counter+=1
                current_time = time.monotonic()
                if (current_time - startTime) > 1 :
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                color = (255, 0, 0)
                self.frame = imgFrame.getCvFrame()
                trackletsData = track.tracklets
                for t in trackletsData:
                    roi = t.roi.denormalize(self.frame.shape[1], self.frame.shape[0])
                    x1 = int(roi.topLeft().x)
                    y1 = int(roi.topLeft().y)
                    x2 = int(roi.bottomRight().x)
                    y2 = int(roi.bottomRight().y)

                    try:
                        label = self.labelMap[t.label]
                    except:
                        label = t.label

                    cv2.putText(self.frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(self.frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(self.frame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                cv2.putText(self.frame, "NN fps: {:.2f}".format(fps), (2, self.frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

                cv2.imshow("tracker", self.frame)
                image_message = self.bridge.cv2_to_imgmsg(self.frame, encoding="bgr8")
                self.publisher_.publish(image_message)
                cv2.waitKey(1) 

        # # Convert frame to ROS Image message and publish
        # image_message = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        # self.publisher_.publish(image_message)
    # def __del__(self):
    #     # Cleanup code when the node is destroyed
    #     if self.device:
    #         del self.device


def main(args=None):
    rclpy.init(args=args)
    depthai_camera_node = DepthAICameraNode()
    rclpy.spin(depthai_camera_node)
    depthai_camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
