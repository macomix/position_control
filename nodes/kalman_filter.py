#!/usr/bin/env python3
import rclpy
from hippo_msgs.msg import RangeMeasurementArray
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped

import numpy as np


class PositionKalmanFilter(Node):

    def __init__(self):
        super().__init__(node_name='position_kalman_filter')

        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         depth=1)

        self.time_last_prediction = self.get_clock().now()

        # TODO Assuming state consists of position x,y,z -> Feel free to add more!
        self.num_states = 3

        # initial state
        # TODO pick a reasonable initial state
        self.x0 = np.zeros((self.num_states, 1))

        # state, this will be updated in Kalman filter algorithm
        self.x = self.x0

        # initial state covariance - how sure are we about the state?
        # TODO initial state covariance is tuning knob
        # dimension: num states x num states
        # matrix needs to be positive definite and symmetric
        self.P0 = np.eye(self.num_states)

        # state covariance, this will be updated in Kalman filter algorithm
        self.P = self.P0

        # process noise covariance - how much noise do we add at each process update step?
        # TODO tuning knob
        # dimension: num states x num states
        # matrix needs to be positive definite and symmetric
        self.Q = np.eye(self.num_states)

        # measurement noise covariance - how much noise does the measurement contain?
        # TODO tuning knob
        # dimnesion: num measurements x num measurements
        # attention, this size is varying! -> Depends on detected Tags
        self.R = np.eye(4)

        # TODO enter tag poses here
        # TODO in the experiment, the tags will not be in these exact positions
        # however, the relative positions between tags will be the same
        self.tag_poses = np.array([[0.7, 3.8, -0.5], [0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        self.position_pub = self.create_publisher(msg_type=PoseStamped,
                                                  topic='position_estimate',
                                                  qos_profile=1)

        self.ranges_sub = self.create_subscription(
            msg_type=RangeMeasurementArray,
            topic='ranges',
            callback=self.on_ranges,
            qos_profile=qos)

        # do process update with 50 Hz
        self.process_update_timer = self.create_timer(
            1 / 50, self.on_process_update_timer)

    def on_ranges(self, ranges_msg: RangeMeasurementArray) -> None:
        # how many tags are detected?
        num_measurements = len(ranges_msg._measurements)

        # if no tags are detected, stop here
        if not num_measurements:
            return

        # initialize some matrices to save the detected tags
        detected_tag_poses = np.zeros((num_measurements, 3))
        distance_measurements = np.zeros((num_measurements, 1))

        for i, measurement in enumerate(ranges_msg.measurements):
            # TODO
            # What tag id was this?
            tag_id = measurement.id

            # Save the according tag position
            detected_tag_poses[i, :] = self.tag_poses[tag_id, :]
            distance_measurements[i, 0] = measurement.range

        # before the measurement update, let's do a process update
        now = self.get_clock().now()
        dt = (now - self.time_last_prediction).nanoseconds * 1e-9
        self.process_update(dt)
        self.time_last_prediction = now

        self.measurement_update(detected_tag_poses, distance_measurements)

    def on_process_update_timer(self):
        # We will do a process update aka prediction with a constant rate
        now = self.get_clock().now()
        dt = (now - self.time_last_prediction).nanoseconds * 1e-9

        self.process_update(dt)
        self.time_last_prediction = now

        # publish the estimated pose with constant rate
        self.publish_pose_msg(state=np.copy(self.x), now=now)

    def measurement_update(self, detected_tags_poses, distance_measurements):
        # TODO
        # What's h(x)?

        # self.x = ....
        # self.P = ....
        pass

    def process_update(self, dt: float):
        # TODO

        # self.x = ....
        # self.P = ....
        pass

    def publish_pose_msg(self, state: np.ndarray, now: rclpy.time.Time) -> None:
        msg = PoseStamped()

        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = state[0, 0]
        msg.pose.position.y = state[1, 0]
        msg.pose.position.z = state[2, 0]

        self.position_pub.publish(msg)


def main():
    rclpy.init()
    node = PositionKalmanFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
