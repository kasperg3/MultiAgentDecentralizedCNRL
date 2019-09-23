//
// Created by kasper on 9/23/19.
//

#ifndef MERGABLEINDUSTRIALROBOTS_URVREPSIM_H
#define MERGABLEINDUSTRIALROBOTS_URVREPSIM_H

#include "rw/rw.hpp"
#include "caros/serial_device_si_proxy.h"
#include "ros/package.h"
#include <iostream>
#include <rw/kinematics.hpp>
#include <rw/invkin/JacobianIKSolver.hpp>
#include <rw/invkin/InvKinSolver.hpp>
#include <geometry_msgs/Transform.h>
#include <std_msgs/Float64.h>

class URVrepSim {
    using Q = rw::math::Q;

public:
    URVrepSim();


private:


};


#endif //MERGABLEINDUSTRIALROBOTS_URVREPSIM_H
