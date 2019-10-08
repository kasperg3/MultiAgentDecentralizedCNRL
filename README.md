# MergableIndustrialRobots

## Connect to UR: 
__1: Specify the ip in the robots interface.__ Settings -> Network -> IP:192.168.100.1

__2: Make your wired connection static and set ip to__ 192.168.100.2

__3: Establish a connection between the robot and the node:__
```
roslaunch caros_universalrobot caros_universalrobot.launch device_ip:=192.168.100.1
```

__4: run your program, eg:__
```
roslaunch caros_universalrobot simple_demo_using_move_ptp.test
```

## Setup V-Rep: 


### Install V-Rep Ros Interface(Requires a catkin workspace):

__1: Install xsltproc and python 2.7 or greater:__
```
    sudo apt-get install xsltproc
```

__2: Make sure to have all ros dependencies:__
```
    rosdep install --from-paths src --ignore-src --rosdistro melodic -y
```

__3: Go to /home/_user_/catkin_ws/ and clone v_rep interface:__
```
    git clone --recursive https://github.com/CoppeliaRobotics/v_repExtRosInterface.git vrep_ros_interface
```

__4: In order to build the packages, navigate to the catkin_ws folder and type: __
```
    export VREP_ROOT=~/path/to/v_rep/folder
    source ~/.bashrc
```

__5: Build the workspace with catkin__
```
    catkin build
```


### How to add a custom service for V-rep

__1: Clone the vrep_ros_interface from https://github.com/kasperg3/vrep_ros_interface into catkin_ws directory (This is a version with a service created for recieving Q's)__
__2: Build vrep_ros_interface__
```
    catkin build vrep_ros_interface && ./home/user/catkin_ws/vrep_ros_interface/install.sh
```
__3: Enjoy a cold beer, it's friday ma dood__


## Install gazebo and moveit
git clone -b melodic-devel https://github.com/ros-industrial/universal_robot.git
 1792  cd ..
 1793  rosdep update
 1794  rosdep install --rosdistro $ROS_DISTRO --ignore-src --from-paths src
 1795  rosdep install --rosdistro melodic --ignore-src --from-paths src
 1796  cd src/
 1797  ls
 1798  catkin build universal_robot
 1799  sudo apt-get install ros-melodic-moveit-core
 1800  catkin build universal_robot
 1801  sudo apt-get install ros-melodic-moveit-kinematics
 1802  catkin build universal_robot
 1803  sudo apt-get install ros-melodic-tf-conversions
 1804  catkin build universal_robot
 1805  rostopic list
 1806  sudo apt-get install ros-melodic-moveit-ros-visualization

