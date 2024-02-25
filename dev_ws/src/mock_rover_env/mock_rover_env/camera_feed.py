from datetime import datetime
import cv2
import rclpy
import numpy as np
from rclpy.time import Time
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from threading import Thread


class CameraFeedPublisher(Node):
    video: cv2.VideoCapture
    frame: cv2.typing.MatLike

    def __init__(self):
        super().__init__('camera_feed_publisher')
        self.publisher = self.create_publisher(
            CompressedImage, '/FRONT_CAM/image_rect_compressed', 10)

        self.video = cv2.VideoCapture(0)
        (_, self.frame) = self.video.read()
        Thread(target=self.updateFeed).start()

        self.create_timer(0.1, self.publish_frame)

    def __del__(self):
        self.video.release()

    def publish_frame(self):
        msg = CompressedImage()
        msg.header.stamp = Time().to_msg()
        msg.header.frame_id = 'FRONT_CAM'
        msg.format = 'jpeg'
        msg.data = np.array(cv2.imencode('.jpg', self.frame)[1]).tostring()
        self.publisher.publish(msg)

    def updateFeed(self):
        while True:
            (_, self.frame) = self.video.read()


def main(args=None):
    rclpy.init(args=args)

    rclpy.spin(CameraFeedPublisher())

    rclpy.shutdown()


if __name__ == '__main__':
    main()
