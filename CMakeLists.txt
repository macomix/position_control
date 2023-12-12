cmake_minimum_required(VERSION 3.5)
project(position_control)
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PIDStamped.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

install(PROGRAMS
  nodes/kalman_filter.py
  nodes/ranges_debugger.py
  nodes/yaw_controller.py
  nodes/position_controller.py
  nodes/pose_setpoint.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
