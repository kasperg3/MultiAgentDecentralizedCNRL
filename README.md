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

## Install RobotiQ Grippers from ros industrial

__1: Clone the directory into catkin_ws/src: __
```
    cd ~/catkin_ws/src
```
```
    git clone https://github.com/ros-industrial/robotiq
```

__2: Install dependencies(these were the missing dependencies for me): __
```
    sudo apt-get install ros-melodic-soem
    sudo apt-get install ros-melodic-socketcan-interface
```
__3: Build:__

```
    catkin build robotiq
```

__4: Running robotiq: __
The hand-e gripper is not supported, so use a robotiq 2f instead. This can be run by following the tutorial on ros:
http://wiki.ros.org/robotiq/Tutorials/Control%20of%20a%202-Finger%20Gripper%20using%20the%20Modbus%20RTU%20protocol%20%28ros%20kinetic%20and%20newer%20releases%29

