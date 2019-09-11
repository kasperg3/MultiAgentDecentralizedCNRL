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


## Install V-Rep Ros Bridge:

__1: Install xsltproc and python 2.7 or greater:__
```
    sudo apt-get install xsltproc 
```

__2: Go to /home/_user_/catkin_ws/ and clone v_rep interface:__
```
git clone --recursive https://github.com/CoppeliaRobotics/v_repExtRosInterface.git vrep_ros_interface
```

__3: Build the workspace with catkin__
```
    catkin_make
```