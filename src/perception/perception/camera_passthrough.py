import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image


class CameraPassthrough(Node):
    """
    Minimal perception entry point.

    Subscribes to the external camera image and republishes it unchanged.
    This node exists to validate wiring, timing, and QoS before adding vision logic.
    """

    def __init__(self):
        super().__init__("camera_passthrough")

        self.declare_parameter(
            "input_topic", "/external_camera/image_raw"
        )
        self.declare_parameter(
            "output_topic", "/perception/image_raw"
        )

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value

        self.publisher = self.create_publisher(
            Image,
            output_topic,
            10,
        )

        self.subscription = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            10,
        )

        self.get_logger().info(
            f"Camera passthrough started: {input_topic} → {output_topic}"
        )

    def image_callback(self, msg: Image):
        # Forward image unchanged
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraPassthrough()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
