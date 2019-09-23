//
// Created by kasper on 9/23/19.
//

#ifndef MERGABLEINDUSTRIALROBOTS_URVREPSIM_H
#define MERGABLEINDUSTRIALROBOTS_URVREPSIM_H

#include <iostream>

#include "ros/node_handle.h"
#include "rw/rw.hpp"
#include "ros/package.h"
#include <rw/kinematics.hpp>
#include <geometry_msgs/Transform.h>
#include <std_msgs/Float64.h>

//Collision detection
#include <rwlibs/proximitystrategies/ProximityStrategyFactory.hpp>

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
private:
    ros::NodeHandle nh;
    rw::models::WorkCell::Ptr wc;
    rw::models::Device::Ptr device;
    rw::kinematics::State state;
    rw::proximity::CollisionDetector::Ptr detector;

};


#endif //MERGABLEINDUSTRIALROBOTS_URVREPSIM_H
