#!/usr/bin/env python3
"""
This node is controlling the position of the BlueROV
based on a given input position.
"""
import numpy as np

import rclpy
from hippo_msgs.msg import ActuatorSetpoint
from rcl_interfaces.msg import SetParametersResult
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
        
        # gains for each direction [k_p, k_i, k_d]
        self.gains_x = np.zeros(3)
        self.gains_y = np.zeros(3)
        self.gains_z = np.zeros(3)
        self.init_params()
        self.add_on_set_parameters_callback(self.on_params_changed)

        self.get_logger().info(f'{self.gains_x}')

        self.current_setpoint = np.zeros(3)
        self.quaternion = np.zeros(4) # w, x, y, z

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

    def init_params(self):
        # load params from config
        self.declare_parameters(namespace='',
                                parameters=[
                                    ("gains_x", rclpy.Parameter.Type.DOUBLE_ARRAY),
                                    ("gains_y", rclpy.Parameter.Type.DOUBLE_ARRAY),
                                    ("gains_z", rclpy.Parameter.Type.DOUBLE_ARRAY)
                                             ])
        
        param = self.get_parameter('gains_x')
        self.gains_x = param.get_parameter_value().double_array_value

        param = self.get_parameter('gains_y')
        self.gains_y = param.get_parameter_value().double_array_value

        param = self.get_parameter('gains_z')
        self.gains_z = param.get_parameter_value().double_array_value

    def on_params_changed(self, params):
        param: rclpy.Parameter
        for param in params:
            self.get_logger().info(f'Try to set [{param.name}] = {param.value}')
            if param.name == 'gains_x':
                self.gains_x = param.get_parameter_value().double_array_value
            elif param.name == 'gains_y':
                self.gains_y = param.get_parameter_value().double_array_value
            elif param.name == 'gains_z':
                self.gains_z = param.get_parameter_value().double_array_value
            else:
                continue
        return SetParametersResult(successful=True, reason='Parameter set')
    
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

        if np.array_equal(self.quaternion, np.zeros(4)):
            self.get_logger().warn(f'nothing yet received from vision_pose_cov')
        
        thrust = self.compute_control_output(current_position, timestamp)

        self.publish_thrust(thrust=thrust, timestamp=timestamp)

    def on_velocity(self, msg: Vector3Stamped):
        # get the estimated velocity from the Kalman filter
        self.velocity = np.array([msg.vector.x, msg.vector.y, msg.vector.z])

    def on_vision_pose(self, msg: PoseWithCovarianceStamped):
        # get the vehicle orientation expressed as quaternion
        q = msg.pose.pose.orientation
        # convert quaternion to euler angles -> dont
        #(roll, pitch, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])

        self.quaternion = np.array([q.w, q.x, q.y, q.z])

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
        """
        The main PID-controller calculations happen in this function.

        Args:
            current_position (np.ndarray): position computed by the Kalman filter
            timestamp (rclpy.time.Time): time of the received data

        Returns:
            np.ndarray: output thrust in x,y,z direction of the robot
        """
        gain = np.array([self.gains_x, self.gains_y, self.gains_z])

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
        thrust = vector3_rotate(vector3=thrust, quaternion=self.quaternion)
        
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
      
def clamp(number, smallest, largest):
    return max(smallest, min(number, largest))

def numpy_to_vector3(array: np.ndarray) -> Vector3:
    # function for safe and convenient type conversion
    arr = array.reshape((1,-1))
    if arr.shape != (1, 3):
        raise ValueError("Size of numpy error does not match a Vector3.")

    rosVector = Vector3()
    rosVector.x = np.float64(array[0])
    rosVector.y = np.float64(array[1])
    rosVector.z = np.float64(array[2])
    
    return rosVector

def quaternion_multiply(quaternion0: np.ndarray, quaternion1: np.ndarray) -> np.ndarray:
    # Hamilton multiplication
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def vector3_rotate(vector3: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    # directly transforms a vector3 using quaternions
    # (much better than euler anglers and faster than rotation matrices)
    q_vec = np.append(0.0, vector3) # 0, x, y, z
    q_inverse = np.concatenate(([quaternion[0]], -quaternion[1:])) # w, -x, -y, -z
    return quaternion_multiply(quaternion, quaternion_multiply(q_vec, q_inverse))[1:]


def main():
    rclpy.init()
    node = PositionControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
