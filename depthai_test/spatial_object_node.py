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
import tf2_ros
from geometry_msgs.msg import TransformStamped


class SpatialObjectNode(Node):
    def __init__(self):
        super().__init__('spatial_camera_node')
        self.declare_parameter('model_path', '/')
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        model_path = self.get_parameter('model_path').get_parameter_value().string_value 
        self.publisher_ = self.create_publisher(Image, 'spatilal_object/image', qos_profile)
        self.bridge = CvBridge()
        # Define sources and outputs
        fullFrameTracking = False
        self.labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.pipeline = dai.Pipeline()
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        spatialDetectionNetwork = self.pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        monoLeft = self.pipeline.create(dai.node.MonoCamera)
        monoRight = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        objectTracker = self.pipeline.create(dai.node.ObjectTracker)

        xoutRgb = self.pipeline.create(dai.node.XLinkOut)
        trackerOut = self.pipeline.create(dai.node.XLinkOut)

        xoutRgb.setStreamName("preview")
        trackerOut.setStreamName("tracklets")

        # Properties
        camRgb.setPreviewSize(300, 300)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camRgb.setFps(40)

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setCamera("left")
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setCamera("right")

        # setting node configs
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Align depth map to the perspective of RGB camera, on which inference is done
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

        spatialDetectionNetwork.setBlobPath(model_path)
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)


        # objectTracker.setDetectionLabelsToTrack([5,15])  # track only person
        objectTracker.setDetectionLabelsToTrack([5])  # track only person
        # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
        objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
        # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
        objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        # Linking
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        # Linking
        camRgb.preview.link(spatialDetectionNetwork.input)
        objectTracker.passthroughTrackerFrame.link(xoutRgb.input)
        objectTracker.out.link(trackerOut.input)

        if fullFrameTracking:
            camRgb.setPreviewKeepAspectRatio(False)
            camRgb.video.link(objectTracker.inputTrackerFrame)
            objectTracker.inputTrackerFrame.setBlocking(False)
            # do not block the pipeline if it's too slow on full frame
            objectTracker.inputTrackerFrame.setQueueSize(2)
        else:
            spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

        spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
        spatialDetectionNetwork.out.link(objectTracker.inputDetections)
        stereo.depth.link(spatialDetectionNetwork.inputDepth)
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
            color = (255, 255, 255)
            while rclpy.ok():
                imgFrame = preview.get()
                track = tracklets.get()
                counter+=1
                current_time = time.monotonic()
                if (current_time - startTime) > 1 :
                    fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

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

                    cv2.putText(self.frame, f"X: {int(t.spatialCoordinates.x)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(self.frame, f"Y: {int(t.spatialCoordinates.y)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(self.frame, f"Z: {int(t.spatialCoordinates.z)} mm", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    try:
                        transform = TransformStamped()
                        transform.header.stamp = self.get_clock().now().to_msg()
                        transform.header.frame_id = 'camera_frame'
                        transform.child_frame_id = f'object_{label}'
                        transform.transform.translation.x = int(t.spatialCoordinates.z)/1000
                        transform.transform.translation.y = -int(t.spatialCoordinates.x)/1000
                        transform.transform.translation.z = int(t.spatialCoordinates.y)/1000
                        transform.transform.rotation.w = 1.0  # Assuming no rotation; adjust as needed
                        self.tf_broadcaster.sendTransform(transform)
                    except:
                        self.print_in_yellow("error broadcasting tf")
                cv2.putText(self.frame, "NN fps: {:.2f}".format(fps), (2, self.frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
                cv2.imshow("tracker", self.frame)
                image_message = self.bridge.cv2_to_imgmsg(self.frame, encoding="bgr8")
                image_message.header.frame_id = "camera_frame"  # Set the frame ID
                image_message.header.stamp = self.get_clock().now().to_msg()  # Set the timestamp
                self.publisher_.publish(image_message)
                cv2.waitKey(1) 

    

def main(args=None):
    rclpy.init(args=args)
    spatial_camera_node = SpatialObjectNode()
    rclpy.spin(spatial_camera_node)
    spatial_camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
