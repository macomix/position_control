#!/usr/bin/env python3
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from hippo_msgs.msg import RangeMeasurement, RangeMeasurementArray
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from tf_transformations import euler_from_quaternion


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

        # process noise covariance - how much noise do we add at each
        # prediction step?
        # TODO tuning knob
        # dimension: num states x num states
        # matrix needs to be positive definite and symmetric
        self.process_noise_position_stddev: float = 0.1
        self.Q = (self.process_noise_position_stddev**2) * np.eye(self.num_states)

        # measurement noise covariance - how much noise does the measurement
        # contain?
        # TODO tuning knob
        # dimension: num measurements x num measurements
        # attention, this size is varying! -> Depends on detected Tags
        # this means you have to create R on the go
        self.range_noise_stddev: float = 0.1

        # TODO: do this different
        self.num_measurements = 0
        #self.curr_measurements = []

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

        self.vision_pose_sub = self.create_subscription(
            msg_type=PoseWithCovarianceStamped,
            topic='vision_pose_cov',
            callback=self.on_vision_pose,
            qos_profile=qos)

        # do prediction step with 50 Hz
        self.process_update_timer = self.create_timer(
            1.0 / 50, self.on_prediction_step_timer)

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
                self.Q = (self.process_noise_position_stddev ** 2) * np.eye(self.num_states)
            else:
                continue
        return SetParametersResult(successful=True, reason='Parameter set')

    def on_ranges(self, ranges_msg: RangeMeasurementArray) -> None:
        # how many tags are detected?
        self.num_measurements = len(ranges_msg._measurements)

        # if no tags are detected, stop here
        if not self.num_measurements:
            return

        # TODO: probably collect measurements and calculate position
        measurements: list[tuple[float, int]] = []

        measurement: RangeMeasurement
        for index, measurement in enumerate(ranges_msg.measurements):
            tag_id = measurement.id
            measured_distance = measurement.range
            # self.get_logger().info(
            #     f'The {index}. element contains the measurement of tag {tag_id} with the '
            #     f'distance of {measured_distance}m')
            # TODO
            measurements.append((measured_distance, tag_id))
            #measurements[index] = [measured_distance, tag_id]

        #self.get_logger().info(f'Measurements {measurements}')

        # before the measurement update, let's do a process update
        now = self.get_clock().now()
        dt = (now - self.time_last_prediction).nanoseconds * 1e-9
        self.prediction(dt)
        self.time_last_prediction = now

        # TODO
        self.measurement_update(measurements)

    def on_vision_pose(self, msg: PoseWithCovarianceStamped):
        # You might want to consider the vehicle's orientation

        # get the vehicle orientation expressed as quaternion
        q = msg.pose.pose.orientation
        # convert quaternion to euler angles
        (roll, pitch, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # TODO

    def on_prediction_step_timer(self):
        # We will do a prediction step with a constant rate
        now = self.get_clock().now()
        dt = (now - self.time_last_prediction).nanoseconds * 1e-9

        self.prediction(dt)
        self.time_last_prediction = now

        # publish the estimated pose with constant rate
        self.publish_pose_msg(state=np.copy(self.state), now=now)

    def measurement_update(self, measurements: list[tuple[float, int]]):
        vehicle_position = np.copy(self.state[0:3, 0])
        
        # diff between estimated measurement based on predicted position and actual measurements
        y = self.get_innovation_y(vehicle_position, measurements)

        # jacobian of observation TODO: probably change the whole calculation of H
        H = self.get_jacobian_H(vehicle_position, measurements)

        # 
        R = (self.range_noise_stddev ** 2) * np.eye(self.num_measurements)

        # compute Kalman gain
        S = H @ self.P @ H.transpose() + R   # innovation covariance
        K = self.P @ H.transpose() @ np.linalg.inv(S)

        
        
        # update state
        state_next = self.state + K @ y
        
        # update covariance
        P_next = (np.eye(self.num_states) - (K @ H)) @ self.P

        #self.get_logger().info(f'y: {y}')
        #self.get_logger().info(f'Next state: {state_next}' f'Current state: {self.state}')

        self.state = state_next
        self.P = P_next

    def get_innovation_y(self, vehicle_position_est, measurements) -> np.ndarray:
        y = np.zeros(self.num_measurements)

        for index, measurement in enumerate(measurements):
            # y = z-z_est
            tag_id = measurement[1]
            y[index] = measurement[0] - np.linalg.norm(self.tag_poses[tag_id] - vehicle_position_est)

        return y.reshape(-1, 1)
    
    def get_jacobian_H(self, vehicle_position, measurements) -> np.ndarray:
        H = np.zeros((self.num_measurements, self.num_states))

        for index, measurement in enumerate(measurements):
            tag_id = measurement[1]
            distance = measurement[0]
            part_der_x = (vehicle_position[0]-self.tag_poses[tag_id, 0])/distance
            part_der_y = (vehicle_position[1]-self.tag_poses[tag_id, 1])/distance
            part_der_z = (vehicle_position[2]-self.tag_poses[tag_id, 2])/distance
            H[index] = np.array([part_der_x, part_der_y, part_der_z])
        
        return H
    
    def get_matrix_A(self, dt: float) -> np.ndarray:
        # jacobian matrix for state-transition model (matrix A or F)
        # TODO: make this actually do something with velocity
        return np.eye(self.num_states)

    def prediction(self, dt: float):
        matrix_A = self.get_matrix_A(dt)
        self.state = matrix_A @ self.state # state_est_next
        self.P = matrix_A @ self.P @ matrix_A.transpose() + self.Q # P_est_next

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
