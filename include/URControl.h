#ifndef URROBOT_H
#define URROBOT_H

#include <rwlibs/proximitystrategies/ProximityStrategyFactory.hpp>
#include <rw/invkin/InvKinSolver.hpp>
#include <rw/math/Transform3D.hpp>
#include <rw/math/Q.hpp>
#include <rw/models/Device.hpp>
#include <rw/kinematics/State.hpp>
#include <rw/models/WorkCell.hpp>
#include <rw/proximity/CollisionDetector.hpp>
#include <rw/loaders/WorkCellLoader.hpp>
#include <rtde_receive_interface.h>
#include <rtde_control_interface.h>
#include <ros/node_handle.h>
#include "ros/package.h"
#include <iostream>

class URControl {
    using Q = rw::math::Q;
private:
    ros::NodeHandle nh;
    rw::models::WorkCell::Ptr wc;
    rw::models::Device::Ptr device;
    rw::kinematics::State state;
    rw::proximity::CollisionDetector::Ptr detector;
    ur_rtde::RTDEReceiveInterface *rtdeReceive;
    ur_rtde::RTDEControlInterface *rtdeControl;

public:

    URControl(std::string);

    Q getQ();
    std::vector<double> qToVector(Q q);
    std::vector<std::vector<double>> transformToVector(rw::math::Transform3D<double>);
    bool moveHome();
    rw::math::Transform3D<> getPose();
    bool moveRelative(Q dq);
    bool checkCollision(Q q);     //Check if given configuration 'q' is in collision:
    bool moveToPose(rw::math::Transform3D<> T_BT_desired);
    void follow_path(std::vector<Q>);
    Eigen::Matrix<double,6,1> computeTaskError(Q qnear, Q qs);
    Q randomConfig();
};


#endif // URROBOT_H
