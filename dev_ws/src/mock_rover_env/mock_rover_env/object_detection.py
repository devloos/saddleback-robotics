from datetime import datetime
from ultralytics import YOLO
import cv2
import rclpy
import numpy as np
from rclpy.time import Time
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from threading import Thread

# Use VideoCapture constructor from cv2
# 0 is used for the default camera, you can change it to a different number if you have multiple cameras
# 1 is used for alternative camera (1/26/2024)
video = cv2.VideoCapture(0)

# Call and store the yolov8 pretrained model
# Load YOLOv8 model with the specified pretrained weights
model = YOLO("yolov8m.pt")

# Opening the classes file in read mode
# Assuming classes.txt contains the names of object classes, one per line
my_file = open("classes.txt", "r")

# Reading the file
data = my_file.read()

# Creating a list of all object classes
classes_list = data.replace('\n', ',').split(
    ",")  # Convert the string of classes to a list

# Closing class file
my_file.close()

# Looping through

# Releasing video capture
video.release()


# Update as of 1/26/2024


class CameraFeedPublisher(Node):
    video: cv2.VideoCapture
    frame: cv2.typing.MatLike

    def __init__(self):
        super().__init__('camera_feed_publisher')
        self.publisher = self.create_publisher(
            CompressedImage, '/FRONT_CAM/image_rect_compressed', 10)

        model = YOLO('best_model(Jan21).pt')

        results = model(source=0, show=True, conf=0.6, save=True)

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
            ret, frame = video.read()  # Read a frame from the camera

            if not ret:
                break

            # Storing model output
            # Run the YOLOv8 model on the current frame using Metal GPU acceleration
            results = model(frame, device="mps")

            # Calling the first element in results
            result = results[0]

            # Creating bounding boxes and converting them to a numpy array
            # Extract bounding boxes from the model output
            bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
            # Extract predicted classes from the model output
            classes = np.array(result.boxes.cls.cpu(), dtype="int")

            # Drawing a bounding box and writing textual classification on top of each box
            for cls, bbox in zip(classes, bboxes):
                (x, y, x2, y2) = bbox

                # Draw a rectangle around the detected object
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)

                # Display the class label above the rectangle
                cv2.putText(
                    frame, classes_list[cls], (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

            # Display the frame with bounding boxes and labels
            cv2.imshow("frame", frame)

            key = cv2.waitKey(1)  # Allows camera feed to remain on

            # Quitting program with the q keyboard input
            if key & 0xFF == ord("q"):
                break


def main(args=None):
    rclpy.init(args=args)

    rclpy.spin(CameraFeedPublisher())

    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
