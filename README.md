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
__REDUCED VERSION__
```
CLONE INTO CATKIN_WS/SRC/
1791  git clone -b melodic-devel https://github.com/ros-industrial/universal_robot.git
 1792  cd ..
 1793  rosdep update
 1794  rosdep install --rosdistro $ROS_DISTRO --ignore-src --from-paths src
 1795  rosdep install --rosdistro melodic --ignore-src --from-paths src
 1827  sudo apt-get install ros-melodic-gazebo-ros
 1829  sudo apt-get install ros-melodic-xacro
 1834  sudo apt-get install ros-melodic-rviz
 1838  sudo apt-get install ros-melodic-moveit-planners-ompl 
 1840  sudo apt-get install ros-melodic-moveit-simple-controller-manager
 1846  sudo apt-get install ros-melodic-moveit-ros-visualization
 1853  sudo apt-get install ros-melodic-robot-state-publisher
 1854  sudo apt-get install ros-melodic-controller-manager
 1851  roslaunch ur5_moveit_config moveit_rviz.launch config:=true
 1852  roslaunch ur5_moveit_config moveit_rviz.launch
 1855  roslaunch ur_gazebo ur5.launch
 1856  rostopic list
```
__FULL VERSION__
```
1791  git clone -b melodic-devel https://github.com/ros-industrial/universal_robot.git
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
 1805  clion
 1806  cd ..
 1807  cd securityLectures/
 1808  ls
 1809  gcc 03_modify_parameter-template.c FFFFF 1
 1810  gcc 03_modify_parameter-template.c -FFFFF -1
 1811  gcc 03_modify_parameter-template.c
 1812  ls
 1813  ./a.out 
 1814  ./a.out hello 1
 1815  ./a.out hellooooooooooooooooo 1
 1816  gcc 03_modify_parameter-template.c
 1817  ./a.out hellooooooooooooooooo 1
 1818  clear
 1819  gcc 03_modify_parameter-template.c -m32 -fno-stack-protector
 1820  gcc -fno-stack-protector -m32 -ggdb -O0 -o 03_modify_parameter-template.c 
 1821  gcc -fno-stack-protector -m32 -ggdb -O0 03_modify_parameter-template.c 
 1822  gcc -ggdb -O0 -fno-stack-protector -no-pie -fno-pic -m32
 1823  gcc -ggdb -O0 -fno-stack-protector -no-pie -fno-pic -m32 03_modify_parameter-template.c 
 1824  la
 1825  ./a.out 
 1826  roslaunch ur_gazebo ur5.launch
 1827  sudo apt-get install ros-melodic-gazebo-ros
 1828  roslaunch ur_gazebo ur5.launch
 1829  sudo apt-get install ros-melodic-xacro
 1830  roslaunch ur_gazebo ur5.launch
 1831  roscore
 1832  roslaunch ur_gazebo ur5.launch
 1833  roslaunch ur5_moveit_config moveit_rviz.launch config:=true
 1834  sudo apt-get install ros-melodic-rviz
 1835  roslaunch ur5_moveit_config moveit_rviz.launch config:=true
 1836  roslaunch ur5_moveit_config moveit_rviz.launch 
 1837  roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch sim:=true
 1838  sudo apt-get install ros-melodic-moveit-planners-ompl 
 1839  roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch sim:=true
 1840  sudo apt-get install ros-melodic-moveit-simple-controller-manager
 1841  roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch sim:=true
 1842  cd catkin_ws/
 1843  catkin build universal_robot
 1844  roslaunch ur_gazebo ur5.launch
 1845  rostopic list
 1846  sudo apt-get install ros-melodic-moveit-ros-visualization
 1847  history
 1848  $ roslaunch ur5_moveit_config moveit_rviz.launch config:=true
 1849  roslaunch ur5_moveit_config moveit_rviz.launch config:=true
 1850  roslaunch ur5_moveit_config moveit_rviz.launch
 1851  roslaunch ur5_moveit_config moveit_rviz.launch config:=true
 1852  roslaunch ur5_moveit_config moveit_rviz.launch
 1853  sudo apt-get install ros-melodic-robot-state-publisher
 1854  sudo apt-get install ros-melodic-controller-manager
 1855  roslaunch ur_gazebo ur5.launch
 1856  rostopic list
 1857  history
```