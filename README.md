# Multi-Agent Decentralised Coordination using CNRL for Industrial Applications


Proof of concept video:
https://www.youtube.com/watch?v=NQcIwR0BVVg&ab_channel=KasperGr%C3%B8ntved

## Connect to UR: 

UR RTDE STUFF TODO

## Prerequisites

__SpinningUp from OpenAIGym:__
```
git clone https://github.com/openai/spinningup.git
cd spinningup
pip3 install -e .
```

__gym-mergablerobots:__

```
git clone https://gitlab.com/kasperg3/mergableindustrialrobots
cd mergableindustrialrobots
pip3 install -e gym-mergablerobots/
```

__Tensorflow:__
```
pip3 install tensorflow
```

Add this to .bashrc:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
```

And source: 
```
cd 
source .bashrc
```






## Setup V-Rep: (DEPRECATED)


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

