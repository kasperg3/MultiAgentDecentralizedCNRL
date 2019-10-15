#ifndef URROBOT_H
#define URROBOT_H

#include <rw/math/Q.hpp>
#include <rw/models/Device.hpp>
#include <rw/kinematics/State.hpp>
#include <rw/models/WorkCell.hpp>
#include <rw/proximity/CollisionDetector.hpp>
#include <rw/loaders/WorkCellLoader.hpp>
//Collision detection
#include <rwlibs/proximitystrategies/ProximityStrategyFactory.hpp>
#include "caros/serial_device_si_proxy.h"
#include "ros/package.h"
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
class URRobot {
    using Q = rw::math::Q;

private:
    ros::NodeHandle nh;
    rw::models::WorkCell::Ptr wc;
    rw::models::Device::Ptr device;
    rw::kinematics::State state;
    caros::SerialDeviceSIProxy* robot;
    //shp:
    rw::proximity::CollisionDetector::Ptr detector;

public:
    URRobot();

    Q getQ();

    bool setQ(Q q);
    bool moveQ(Q dq);
    bool moveHome();
    bool checkCollision(Q q);     //Check if given configuration 'q' is in collision:
    bool moveToPose(rw::math::Transform3D<> T_BT_desired);
    bool moveToPose(rw::math::RPY<> RPY_desired, rw::math::Vector3D<> displacement_desired);
    bool moveToPose(double arr[]); //State-vector: [X,Y,Z,R,P,Y]
    void follow_path(std::vector<Q>);
    rw::math::Transform3D<> getPose();

    Eigen::Matrix<double,6,1> computeTaskError(Q qnear, Q qs);
    Q randomConfig();

    Q inverseKin();



};


#endif // URROBOT_H
