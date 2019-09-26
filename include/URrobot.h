#ifndef URROBOT_H
#define URROBOT_H

#include "rw/rw.hpp"
#include <rwlibs/proximitystrategies/ProximityStrategyFactory.hpp>
#include <rw/invkin/InvKinSolver.hpp>
#include <rw/math/Transform3D.hpp>

#include "caros/serial_device_si_proxy.h"
#include "ros/package.h"
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

using namespace std;

using namespace rw;
using namespace rw::models;
using namespace rw::kinematics;
using namespace rw::proximity;
using namespace rw::common;
using namespace rw::math;

using namespace rwlibs::proximitystrategies;


int myfunc(int a, int b);

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
    bool goToZeroPos();
    bool checkCollision(Q q);     //Check if given configuration 'q' is in collision:
    bool moveToPose(rw::math::Transform3D<> T_BT_desired);
    bool moveToPose(RPY<> RPY_desired, Vector3D<> displacement_desired);
    bool moveToPose(double arr[]); //State-vector: [X,Y,Z,R,P,Y]
    void follow_path(std::vector<Q>);
    Transform3D<> getPose();

    Eigen::Matrix<double,6,1> computeTaskError(Q qnear, Q qs);
    Q randomConfig();


};


#endif // URROBOT_H
