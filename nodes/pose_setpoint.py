#!/usr/bin/env python3
"""
This node publishes a setpoint for the position and yaw controller.
"""
import rclpy
from hippo_msgs.msg import Float64Stamped
from rclpy.node import Node
import math
import numpy as np

from geometry_msgs.msg import Vector3Stamped

class PoseSetpointNode(Node):

    def __init__(self):
        super().__init__(node_name='pose_setpoint_publisher')

        self.start_time = self.get_clock().now()

        # change these parameters to adjust the setpoints
        self.duration = 30.0  # in seconds

        self.posistion_setpoint_pub = self.create_publisher(msg_type=Vector3Stamped,
                                                        topic='position_setpoint',
                                                        qos_profile=1)

        self.yaw_setpoint_pub = self.create_publisher(msg_type=Float64Stamped,
                                                        topic='yaw_setpoint',
                                                        qos_profile=1)

        self.timer = self.create_timer(timer_period_sec=1 / 50,
                                       callback=self.on_timer)

    def on_timer(self) -> None:
        # change this for other setpoint functions
        now = self.get_clock().now()
        time = self.start_time - now

        position = np.array([1.0,2.0, -0.3])

        function = 0
        match function:
            case 0:
                # square sine
                i = time.nanoseconds * 1e-9 % (self.duration * 2)
                if i > (self.duration):
                    position =  np.array([1.0,2.0, -0.3])
                else:
                    position =  np.array([1.5,1.0, -0.3])
            case 1:
                # something
                position =  position
            case _:
                position =  position

        yaw = np.deg2rad(90)

        now = self.get_clock().now()
        self.publish_setpoint_position(setpoint=position, now=now)
        self.publish_setpoint_yaw(setpoint=yaw, now=now)

    def publish_setpoint_position(self, setpoint: np.ndarray, now: rclpy.time.Time) -> None:
        msg = Vector3Stamped()

        msg.vector.x = setpoint[0]
        msg.vector.y = setpoint[1]
        msg.vector.z = setpoint[2]

        msg.header.stamp = self.get_clock().now().to_msg()
        self.posistion_setpoint_pub.publish(msg)

    def publish_setpoint_yaw(self, setpoint: float, now: rclpy.time.Time) -> None:
        msg = Float64Stamped()

        msg.data = setpoint

        msg.header.stamp = self.get_clock().now().to_msg()
        self.yaw_setpoint_pub.publish(msg)


def main():
    rclpy.init()
    node = PoseSetpointNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
