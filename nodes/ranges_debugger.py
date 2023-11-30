#!/usr/bin/env python3
import rclpy
from hippo_msgs.msg import RangeMeasurementArray
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32MultiArray


class RangesDebugger(Node):

    def __init__(self):
        super().__init__(node_name='range_debugger')

        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         depth=1)

        self.ranges_debug_pub = self.create_publisher(
            msg_type=Float32MultiArray, topic='debug', qos_profile=1)
        self.ranges_sub = self.create_subscription(
            msg_type=RangeMeasurementArray,
            topic='ranges',
            callback=self.on_ranges,
            qos_profile=qos)

    def on_ranges(self, ranges_msg: RangeMeasurementArray) -> None:
        num_measurements = len(ranges_msg.measurements)

        msg = Float32MultiArray()
        msg.data = [0.0] * 4

        if num_measurements:
            for i, tag in enumerate(ranges_msg.measurements):
                msg.data[tag.id] = tag.range

        self.ranges_debug_pub.publish(msg)


def main():
    rclpy.init()
    node = RangesDebugger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
