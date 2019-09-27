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

__1: add your custom serviec file(.srv) to the vrep_ros_interface package:__
```
    mkdir ~/catkin_ws/src/vrep_ros_interface/srv
    cp ~/catkin_ws/src/mergableindustrialrobots/srv/moveRobot.srv ~/catkin_ws/src/vrep_ros_interface/srv/
    cp ~/catkin_ws/src/mergableindustrialrobots/srv/moveRobot.srv ~/catkin_ws/src/vrep_ros_interface/
```

__2: Edit the Meta service tag__
```
    echo "vrep_ros_interface/moveRobot" >> ~/catkin_ws/src/vrep_ros_interface/meta/services.txt
```

__3.1: Add this to vrep_ros_interface CMakelist.txt__
```
    add_service_files(FILES moveRobot.srv)

    generate_messages(
    DEPENDENCIES
    std_msgs
    )
```

__3.2: and to project.xml__
```
    <depend>roslib</depend>
```


