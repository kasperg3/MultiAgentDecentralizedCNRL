//
// Created by kasper on 9/23/19.
//

#ifndef MERGABLEINDUSTRIALROBOTS_URVREPSIM_H
#define MERGABLEINDUSTRIALROBOTS_URVREPSIM_H

#include <iostream>

#include "ros/node_handle.h"
#include "ros/package.h"
#include "ros/publisher.h"
#include "rw/rw.hpp"
#include <rw/kinematics.hpp>
#include <std_msgs/Float32MultiArray.h>
//Collision detection
#include <rwlibs/proximitystrategies/ProximityStrategyFactory.hpp>
#include "mergable_industrial_robots/moveRobot.h"

class URVrepSim {
    using Q = rw::math::Q;

public:
    URVrepSim();
    Q getQ();
    bool setQ(Q);
    bool moveQ(Q);
    bool moveHome();
    rw::kinematics::State getState();
    rw::models::Device::Ptr getDevice();
    void publishQ(Q,  ros::Publisher);
private:
    bool callVrepService(Q);
    ros::ServiceClient client;
    ros::NodeHandle nh;
    rw::models::WorkCell::Ptr wc;
    rw::models::Device::Ptr device;
    rw::kinematics::State state;
    rw::proximity::CollisionDetector::Ptr detector;
    Q defaultQ;

};


#endif //MERGABLEINDUSTRIALROBOTS_URVREPSIM_H
