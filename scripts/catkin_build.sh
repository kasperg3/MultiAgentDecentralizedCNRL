#bin/bash
echo "Cleaning workspace"
catkin clean -y

echo "Building packages in the correct order \n"

catkin build vrep_ros_interface
catkin build caros_control_msgs
catkin build caros_common
catkin build caros_control

catkin build