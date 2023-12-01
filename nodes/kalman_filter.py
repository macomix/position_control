#!/usr/bin/env python3
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from hippo_msgs.msg import RangeMeasurement, RangeMeasurementArray
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy


class PositionKalmanFilter(Node):

    def __init__(self):
        super().__init__(node_name='position_kalman_filter')

        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         depth=1)

        self.init_params()

        self.time_last_prediction = self.get_clock().now()

        # TODO Assuming state consists of position x,y,z -> Feel free to add
        # more!
        self.num_states = 3

        # initial state
        # TODO pick a reasonable initial state
        self.x0 = np.zeros((self.num_states, 1))

        # state, this will be updated in Kalman filter algorithm
        self.state = np.copy(self.x0)

        # initial state covariance - how sure are we about the state?
        # TODO initial state covariance is tuning knob
        # dimension: num states x num states
        # matrix needs to be positive definite and symmetric
        self.P0 = 0.1 * np.eye(self.num_states)

        # state covariance, this will be updated in Kalman filter algorithm
        self.P = self.P0

        # process noise covariance - how much noise do we add at each process
        # update step?
        # TODO tuning knob
        # dimension: num states x num states
        # matrix needs to be positive definite and symmetric
        self.process_noise_position_stddev = 0.1
        self.Q = (self.range_noise_stddev**2) * np.eye(self.num_states)

        # measurement noise covariance - how much noise does the measurement
        # contain?
        # TODO tuning knob
        self.range_noise_stddev = 0.1
        # dimnesion: num measurements x num measurements
        # attention, this size is varying! -> Depends on detected Tags

        # TODO enter tag poses here
        # TODO in the experiment, the tags will not be in these exact positions
        # however, the relative positions between tags will be the same
        self.tag_poses = np.array([[0.7, 3.8, -0.5], [1.3, 3.8, -0.5],
                                   [0.7, 3.8, -0.9], [1.3, 3.8, -0.9]])

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
            1.0 / 50, self.on_process_update_timer)

    def init_params(self):
        self.declare_parameters(namespace='',
                                parameters=[('range_noise_stddev',
                                             rclpy.Parameter.Type.DOUBLE),
                                            ('process_noise_position_stddev',
                                             rclpy.Parameter.Type.DOUBLE)])
        param = self.get_parameter('range_noise_stddev')
        self.get_logger().info(f'{param.name}={param.value}')
        self.range_noise_stddev = param.value

        param = self.get_parameter('process_noise_position_stddev')
        self.get_logger().info(f'{param.name}={param.value}')
        self.process_noise_position_stddev = param.value

    def on_params_changed(self, params):
        param: rclpy.Parameter
        for param in params:
            self.get_logger().info(f'Try to set [{param.name}] = {param.value}')
            if param.name == 'range_noise_stddev':
                self.range_noise_stddev = param.value
            elif param.name == 'process_noise_position_stddev':
                self.process_noise_position_stddev = param.value
                self.Q = (self.process_noise_position_stddev**2) * np.eye(
                    self.num_states)
            else:
                continue
        return SetParametersResult(successful=True, reason='Parameter set')

    def on_ranges(self, ranges_msg: RangeMeasurementArray) -> None:
        # how many tags are detected?
        num_measurements = len(ranges_msg._measurements)

        # if no tags are detected, stop here
        if not num_measurements:
            return

        # initialize some matrices to save the detected tags
        detected_tag_poses = np.zeros((num_measurements, 3))
        distance_measurements = np.zeros((num_measurements, 1))

        measurement: RangeMeasurement
        for index, measurement in enumerate(ranges_msg.measurements):
            # TODO
            # What tag id was this?
            tag_id = int(measurement.id)

            # add this tag's information to list of detected tags
            detected_tag_poses[index, :] = self.tag_poses[tag_id, :]
            distance_measurements[index, 0] = measurement.range

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
        self.publish_pose_msg(state=np.copy(self.state), now=now)

    def measurement_update(self, tag_positions, distance_measurements):
        # TODO
        vehicle_position = np.copy(self.state[0:3, 0])

        num_measurements = np.shape(distance_measurements)[0]
        R = (self.range_noise_stddev**2) * np.eye(num_measurements)

        z_est = np.zeros((num_measurements, 1))
        H = np.zeros((num_measurements, self.num_states))

        for index, tag_position in enumerate(tag_positions):
            distance_vector = tag_position - vehicle_position
            # this is the distance that we would have expected based on the
            # current estimate of the vehicle position.
            dist_est = np.linalg.norm(distance_vector)
            z_est[index, 0] = dist_est

            # dh / dx = 1/2 * (dist ** 2)^(-1/2) * (2 * (x1 - t1) * 1)
            H[index, 0:3] = [(vehicle_position[0] - tag_position[0] / dist_est),
                             (vehicle_position[1] - tag_position[1] / dist_est),
                             (vehicle_position[2] - tag_position[2] / dist_est)]

        y = distance_measurements - z_est

        # tmp = np.matmul(np.matmul(H, self.P), H.transpose()) + R
        # K = np.matmul(np.matmul(self.P, H.transpose()), np.linalg.inv(tmp))
        # @nbauschmann why not the notation below?
        tmp = H @ self.P @ H.transpose() + R
        K = self.P @ H.transpose() @ np.linalg.inv(tmp)

        # update state
        self.state = self.state + K @ y
        # update covariance
        P_tmp = np.eye(self.num_states) - K @ H
        self.P = P_tmp @ self.P

    def process_update(self, dt: float):
        # TODO
        A = np.eye(3)

        # self.state = np.matmul(np.copy(A), self.state)
        self.state = A @ self.state
        # self.P = np.matmul(np.matmul(np.copy(A), np.copy(self.P)),
        #                   (np.transpose(np.copy(A)))) + self.Q
        self.P = A @ self.P @ A.transpose() + self.Q

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
