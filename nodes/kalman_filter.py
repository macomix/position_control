#!/usr/bin/env python3
"""
Extended Kalman filter
"""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Vector3Stamped
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

        # TODO: tuning params we want to change at runtime
        # change in config file or better: call "rqt"
        # in terminal to dynamically change params at runtime
        self.process_noise_position_stddev: float
        self.range_noise_stddev: float
        self.init_params()
        self.add_on_set_parameters_callback(self.on_params_changed)

        self.time_last_prediction = self.get_clock().now()

        # state (dimension: n):
        # position x,y,z
        # velocity x,y,z
        self.num_states = 6 

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

        # dimension: num states x num states
        # matrix needs to be positive definite and symmetric
        self.Q = (self.process_noise_position_stddev**2) * np.eye(self.num_states)
        
        # TODO: maybe dont save this here
        self.num_measurements = 0 # dimension: m

        # TODO enter tag poses here
        # however, the relative positions between tags will be the same
        # self.tag_poses = np.array([[0.7, 3.8, -0.5], [1.3, 3.8, -0.5], 
        #                            [0.7, 3.8, -0.9], [1.3, 3.8, -0.9]])
        
        self.offsetXYZ = np.array([0.634, 3.8, -0.498])
        self.tag_poses = np.array([[self.offsetXYZ[0], self.offsetXYZ[1], self.offsetXYZ[2]], 
                                   [self.offsetXYZ[0] + 0.635, self.offsetXYZ[1], self.offsetXYZ[2]], 
                                   [self.offsetXYZ[0], self.offsetXYZ[1], -0.89], 
                                   [self.offsetXYZ[0] + 0.635, self.offsetXYZ[1], -0.89]])

        # publisher
        self.position_pub = self.create_publisher(msg_type=PoseStamped,
                                                  topic='position_estimate',
                                                  qos_profile=1)
        
        self.velocity_pub = self.create_publisher(msg_type=Vector3Stamped,
                                                  topic='velocity_estimate',
                                                  qos_profile=1)

        # subscriber
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
        self.range_noise_stddev = param.get_parameter_value().double_value

        param = self.get_parameter('process_noise_position_stddev')
        self.get_logger().info(f'{param.name}={param.value}')
        self.process_noise_position_stddev = param.get_parameter_value().double_value

    def on_params_changed(self, params):
        param: rclpy.Parameter
        for param in params:
            self.get_logger().info(f'Try to set [{param.name}] = {param.value}')
            if param.name == 'range_noise_stddev':
                self.range_noise_stddev = param.get_parameter_value().double_value
            elif param.name == 'process_noise_position_stddev':
                self.process_noise_position_stddev = param.get_parameter_value().double_value
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

        #self.get_logger().info(f'dt: {dt}, gain: {self.process_noise_position_stddev**2}')
        self.Q = self.get_matrix_Q(dt) # update process noise

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
        self.get_logger().debug(f'Hi')

        now = self.get_clock().now()
        dt = (now - self.time_last_prediction).nanoseconds * 1e-9

        self.prediction(dt)
        self.time_last_prediction = now

        # publish the estimated velocity
        self.publish_velocity_msg(state=np.copy(self.state), now=now)

        # publish the estimated pose with constant rate
        self.publish_pose_msg(state=np.copy(self.state), now=now)

    def measurement_update(self, measurements: list[tuple[float, int]]):
        vehicle_position = np.copy(self.state[0:3, 0])
        
        # diff between estimated measurement based on predicted position and actual measurements
        y = self.get_innovation_y(vehicle_position, measurements)

        # jacobian of observation TODO: probably change the whole calculation of H
        H = self.get_jacobian_H(vehicle_position, measurements)

        # covariance of the observation noise (dim: [m, m])
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
        """Calculates the difference between the prediction and the observation.

        Args:
            vehicle_position_est (_type_): x,y,z position of the robot
            measurements (_type_): all of the measurements in an array

        Returns:
            np.ndarray: innovation y (dim: [m, 1])
        """
        y = np.zeros(self.num_measurements)

        for index, measurement in enumerate(measurements):
            # y = z-z_estimated
            tag_id = measurement[1]
            y[index] = measurement[0] - np.linalg.norm(self.tag_poses[tag_id] - vehicle_position_est)

        return y.reshape(-1, 1)
    
    def get_jacobian_H(self, vehicle_position, measurements) -> np.ndarray:
        """Observation model, which estimates measurements based on the estimated state.

        Args:
            vehicle_position (_type_): x,y,z position of the robot
            measurements (_type_): all of the measurements in an array

        Returns:
            np.ndarray: observation matrix (dim: [m,n])
        """
        
        H = np.zeros((self.num_measurements, self.num_states))
        
        for index, measurement in enumerate(measurements):
            tag_id = measurement[1]
            distance = np.linalg.norm(self.tag_poses[tag_id] - vehicle_position)
            part_der_x = (vehicle_position[0]-self.tag_poses[tag_id, 0])/distance
            part_der_y = (vehicle_position[1]-self.tag_poses[tag_id, 1])/distance
            part_der_z = (vehicle_position[2]-self.tag_poses[tag_id, 2])/distance
            H[index] = np.array([part_der_x, part_der_y, part_der_z, 0, 0, 0])
        
        return H
    
    def get_matrix_A(self, dt: float) -> np.ndarray:
        """This function creates the jacobian matrix 
        for the state-transition model (matrix A or F).

        Args:
            dt (float): delta time

        Returns:
            np.ndarray: state-transition matrix (dim: [n,n])
        """

        # this creates a matrix eye(num_states) but also with dt
        # to calculate e.g. x_next=x + dt * dx
        A = np.concatenate((np.eye(3), dt* np.eye(3)), axis=1)
        B = np.concatenate((np.zeros((3,3)), np.eye(3)), axis=1)
        return np.concatenate((A,B), axis=0)

    def get_matrix_Q(self, dt: float) -> np.ndarray:
        """process noise covariance - How much noise do we add at each prediction step? 
        -> The higher dt the more time between measurements
        which results in a higher uncertainty.
        source 1: Wikipedia
        There are multiple ways to calculate Q:
            1) piecewise white noise model
            2) continuous white noise model
            3) ...

        Args:
            dt (float): delta time

        Returns:
            np.ndarray: covariance matrix of the process noise (dim: [n, n])
        """
        # in this case: piecewise white noise model
        q0 = (dt**4)/4
        q1 = (dt**3)/2
        q2 = dt**2
        # NOTE: Q has to be positive definite and symmetric
        Q = np.array([[q0, 0, 0, q1, 0 ,0],
                      [0, q0, 0, 0, q1, 0],
                      [0, 0, q0, 0, 0, q1],
                      [q1, 0, 0, q2, 0, 0],
                      [0, q1, 0, 0, q2, 0],
                      [0, 0, q1, 0, 0, q2]])
        return (self.process_noise_position_stddev**2) * Q

    def prediction(self, dt: float):
        """This function is the first part of the Kalman-Filter.
        It makes a prediction on the current position of the robot.

        Args:
            dt (float): delta time
        """

        matrix_A = self.get_matrix_A(dt)
        self.state = matrix_A @ self.state # estimated state
        self.P = matrix_A @ self.P @ matrix_A.transpose() + self.Q # estimated covariance matrix

    def publish_pose_msg(self, state: np.ndarray, 
                         now: rclpy.time.Time) -> None: # type: ignore
        msg = PoseStamped()

        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = state[0, 0]
        msg.pose.position.y = state[1, 0]
        msg.pose.position.z = state[2, 0]

        self.position_pub.publish(msg)

    def publish_velocity_msg(self, state: np.ndarray, 
                             now: rclpy.time.Time) -> None: # type: ignore
        msg = Vector3Stamped()

        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "map"
        msg.vector.x = state[3, 0]
        msg.vector.y = state[4, 0]
        msg.vector.z = state[5, 0]

        self.velocity_pub.publish(msg)

def main():
    rclpy.init()
    node = PositionKalmanFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
