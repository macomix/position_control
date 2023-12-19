#!/usr/bin/env python3
"""
This node is controlling the position of the BlueROV
based on a given input position.
"""
import numpy as np

from sympy import euler

import rclpy
from hippo_msgs.msg import ActuatorSetpoint
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped, Vector3Stamped, Vector3, PoseWithCovarianceStamped

from tf_transformations import euler_from_quaternion

from position_control.msg import PIDStamped

class PositionControlNode(Node):

    def __init__(self):
        super().__init__(node_name='position_controller')
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         depth=1)

        self.current_setpoint = np.zeros(3)
        self.rotation_matrix = np.eye(3)

        self.velocity = np.zeros(3)
        self.setpoint_velocity = np.zeros(3)

        self.error_integral = np.zeros(3)
        self.last_time = self.get_clock().now().nanoseconds * 10e-9

        # publisher
        self.thrust_pub = self.create_publisher(msg_type=ActuatorSetpoint,
                                                topic='thrust_setpoint',
                                                qos_profile=1)

        # debug publisher
        self.pid_debug_pub = self.create_publisher(msg_type=PIDStamped,
                                                topic='pid_gain',
                                                qos_profile=1)

        # subscriber
        self.setpoint_sub = self.create_subscription(msg_type=Vector3Stamped,
                                                     topic='position_setpoint',
                                                     callback=self.on_setpoint,
                                                     qos_profile=1)
        
        self.setpoint_velocity_sub = self.create_subscription(msg_type=Vector3Stamped,
                                                     topic='velocity_setpoint',
                                                     callback=self.on_setpoint_velocity,
                                                     qos_profile=1)
        
        self.position_sub = self.create_subscription(msg_type=PoseStamped,
                                                  topic='position_estimate',
                                                  callback=self.on_position,
                                                  qos_profile=1)
        
        self.velocity_sub = self.create_subscription(msg_type=Vector3Stamped,
                                                  topic='velocity_estimate',
                                                  callback=self.on_velocity,
                                                  qos_profile=1)
        
        self.vision_pose_sub = self.create_subscription(
                                                    msg_type=PoseWithCovarianceStamped,
                                                    topic='vision_pose_cov',
                                                    callback=self.on_vision_pose,
                                                    qos_profile=qos)

    def on_setpoint(self, setpoint_msg: Vector3Stamped):
        # on setpoint message received save data
        self.current_setpoint = np.array([setpoint_msg.vector.x,setpoint_msg.vector.y,setpoint_msg.vector.z])

    def on_setpoint_velocity(self, setpoint_msg: Vector3Stamped):
        # on setpoint velocity message received save data
        self.setpoint_velocity = np.array([setpoint_msg.vector.x,setpoint_msg.vector.y,setpoint_msg.vector.z])

    def on_position(self, position_msg: PoseStamped):
        # on position message received calculate the thrust
        curr_pos_x = position_msg.pose.position.x
        curr_pos_y = position_msg.pose.position.y
        curr_pos_z = position_msg.pose.position.z

        current_position = np.array([curr_pos_x,curr_pos_y,curr_pos_z])

        # either set the timestamp to the current time or set it to the
        # stamp of `depth_msg` because the control output corresponds to this
        # point in time. Both choices are meaningful.
        # option 1:
        # now = self.get_clock().now()
        # option 2:
        timestamp = rclpy.time.Time.from_msg(position_msg.header.stamp) # type: ignore

        thrust = self.compute_control_output(current_position, timestamp)

        self.publish_thrust(thrust=thrust, timestamp=timestamp)

    def on_velocity(self, msg: Vector3Stamped):
        # get the estimated velocity from the Kalman filter
        self.velocity = np.array([msg.vector.x, msg.vector.y, msg.vector.z])

    def on_vision_pose(self, msg: PoseWithCovarianceStamped):
        # get the vehicle orientation expressed as quaternion
        q = msg.pose.pose.orientation
        # convert quaternion to euler angles
        (roll, pitch, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])

        #self.rotation_matrix = euler_rotation_matrix(0, 0, yaw)
        self.rotation_matrix = quaternion_rotation_matrix(q.w, q.x, q.y, q.z)

    def publish_thrust(self, thrust: np.ndarray, 
                       timestamp: rclpy.time.Time) -> None: # type: ignore
        msg = ActuatorSetpoint()
        # we want to set thrust in every direction.
        msg.ignore_x = False
        msg.ignore_y = False
        msg.ignore_z = False

        msg.x = np.float64(thrust[0])
        msg.y = np.float64(thrust[1])
        msg.z = np.float64(thrust[2])

        # add a time stamp
        msg.header.stamp = timestamp.to_msg()

        self.thrust_pub.publish(msg)

    def compute_control_output(self, current_position: np.ndarray, 
                               timestamp: rclpy.time.Time) -> np.ndarray: # type: ignore
        # gains for each direction [k_p, k_i, k_d]
        # TODO: z should depend on direction because of bouyancy
        gain_x = np.array([0.0, 0.0, 0.0]) # ([1.0, 0.05, 3.0])
        gain_y = np.array([0.0, 0.0, 0.0]) # ([1.0, 0.05, 3.0])
        gain_z = np.array([0.0, 0.0, 0.0]) # ([1.0, 0.05, 3.0])
        gain = np.array([gain_x, gain_y, gain_z])

        # safe area for the robot to operate [min, max]
        safe_space_x = np.array([0.1, 1.9])
        safe_space_y = np.array([0.1, 3])
        safe_space_z = np.array([-0.8, -0.1])
        safe_space = np.array([safe_space_x, safe_space_y, safe_space_z])

        dt = self.get_clock().now().nanoseconds * 10e-9 - self.last_time

        # sort out all abnormal dt
        if dt > 1:
            dt = 0.0

        thrust = np.zeros(3)
        error = self.current_setpoint - current_position
        derivative_error = self.setpoint_velocity - self.velocity

        # calculate thrust for each direction
        # loop because we have to check if the robot is inside the safe area
        for i in range(3):
            pos = current_position[i]
            
            # only add thrust in the safe operating area [min, max]
            if pos < safe_space[i ,1] and pos > safe_space[i, 0]:
                if np.abs(error[i]) < 0.05:
                    self.error_integral[i] = self.error_integral[i] + dt * error[i]
                else:
                    self.error_integral[i] = 0

                # final PID calculation!
                thrust[i] = gain[i, 0] * error[i] + gain[i, 1] * self.error_integral[i] + gain[i, 2] * derivative_error[i]

                # clamp thrust to the range of [-1, 1]
                #thrust[i] = clamp(thrust[i], -1.0, 1.0)
            else:
                self.error_integral[i] = 0.0
                thrust[i] = 0.0

        # convert to local space of the robot
        thrust = np.matmul(self.rotation_matrix.transpose(), thrust.reshape((-1,1)))
        
        # publish some information for debugging and documentation
        self.publish_pid_info(gain, error, self.error_integral, derivative_error, timestamp)

        self.last_time = self.get_clock().now().nanoseconds * 10e-9
        return thrust.reshape(-1)

    def publish_pid_info(self, gains: np.ndarray, error: np.ndarray, 
                         i_error: np.ndarray, d_error:np.ndarray, 
                         timestamp: rclpy.time.Time): # type: ignore
        msg = PIDStamped()

        msg.gain_p = numpy_to_vector3(np.array([gains[0, 0],gains[1, 0],gains[2, 0]]))
        msg.gain_i = numpy_to_vector3(np.array([gains[0, 1],gains[1, 1],gains[2, 1]]))
        msg.gain_d = numpy_to_vector3(np.array([gains[0, 2],gains[1, 2],gains[2, 2]]))

        msg.error = numpy_to_vector3(error)
        msg.error_integral = numpy_to_vector3(i_error)
        msg.error_derivative = numpy_to_vector3(d_error)

        msg.header.stamp = timestamp.to_msg()
        
        self.pid_debug_pub.publish(msg)
        
# TODO: put utility functions into separate file
def clamp(number, smallest, largest):
    return max(smallest, min(number, largest))

def numpy_to_vector3(array: np.ndarray) -> Vector3:
    arr = array.reshape((1,-1))
    if arr.shape != (1, 3):
        raise ValueError("Size of numpy error does not match a Vector3.")

    rosVector = Vector3()
    rosVector.x = np.float64(array[0])
    rosVector.y = np.float64(array[1])
    rosVector.z = np.float64(array[2])
    
    return rosVector

def quaternion_rotation_matrix(w, x, y, z):
     
    # First row of the rotation matrix
    r00 = 1-2*y*y-2*z*z
    r01 = 2*x*y-2*z*w
    r02 = 2*x*z+2*y*w
     
    # Second row of the rotation matrix
    r10 = 2*x*y+2*z*w
    r11 = 1-2*x*x-2*z*z
    r12 = 2*y*z-2*x*w
     
    # Third row of the rotation matrix
    r20 = 2*x*z-2*y*w
    r21 = 2*y*z+2*x*w
    r22 = 1-2*x*x-2*y*y
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

def euler_rotation_matrix(roll, pitch, yaw):
    # radiant
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    
    Rxy = np.matmul(Rx, Ry)
    R = np.matmul(Rxy, Rz)

    return R


def main():
    rclpy.init()
    node = PositionControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
