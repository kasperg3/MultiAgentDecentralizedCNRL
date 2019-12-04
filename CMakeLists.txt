cmake_minimum_required(VERSION 2.8.3)
project(mergable_industrial_robots)

## Add support for C++11, supported in ROS Kinetic and newer
set (CMAKE_CXX_STANDARD 17)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        roslib
        caros_common_msgs
        caros_control_msgs
        caros_common
        caros_control
        std_msgs
        message_generation
        )

add_service_files(
        FILES
        moveRobot.srv
)

generate_messages(
        DEPENDENCIES
        std_msgs
)

###################################
##         RobWork setup         ##
###################################
# Set the RobWork root (edit in .bashrc if necessary)
SET(RW_ROOT $ENV{RW_ROOT})

# Use RobWork
MESSAGE("-- RobWork root: " ${RW_ROOT})
SET(RobWork_DIR ${RW_ROOT}/cmake)
FIND_PACKAGE(RobWork REQUIRED)
INCLUDE_DIRECTORIES( ${ROBWORK_INCLUDE_DIRS} )
LINK_DIRECTORIES( ${ROBWORK_LIBRARY_DIRS} )

###################################
##         PCL setup             ##
###################################
#find_package(PCL 1.3 REQUIRED COMPONENTS)# common io visualization)
#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})


###################################
## catkin specific configuration ##
###################################
catkin_package(
        CATKIN_DEPENDS message_runtime
        #  INCLUDE_DIRS include
        #  LIBRARIES rovi2_rob
        #  CATKIN_DEPENDS roscpp
        #  DEPENDS system_lib
)

###################################
##         UR_RTDE setup         ##
###################################

find_package(ur_rtde REQUIRED)

###################################
##           NEAT setup          ##
###################################
add_subdirectory(lib/NEAT/)
include_directories(lib/NEAT/include)

###########
## Build ##
###########

include_directories(include
        ${catkin_INCLUDE_DIRS}
)

set(SOURCES
        src/main.cpp
        src/URVrep.cpp include/URVrep.h
        src/URControl.cpp include/URControl.h
        src/TTTRL.cpp include/TTTRL.h
        )

add_executable(mergableIndustrialRobots ${SOURCES})
target_link_libraries(mergableIndustrialRobots PRIVATE ur_rtde::rtde ${catkin_LIBRARIES} ${ROBWORK_LIBRARIES} ${ROS_LIBRARIES} ${PYTHON_LIBRARIES})
add_dependencies(mergableIndustrialRobots ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})