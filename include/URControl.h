#ifndef URROBOT_H
#define URROBOT_H

#include <rwlibs/proximitystrategies/ProximityStrategyFactory.hpp>
#include <rw/math/Transform3D.hpp>
#include <rw/math/Q.hpp>
#include <rw/models/Device.hpp>
#include <rw/kinematics/State.hpp>
#include <rw/models/WorkCell.hpp>
#include <rw/proximity/CollisionDetector.hpp>
#include <rw/loaders/WorkCellLoader.hpp>
#include <rtde_receive_interface.h>
#include <rtde_io_interface.h>
#include <rtde_control_interface.h>
#include <ros/node_handle.h>
#include "ros/package.h"
#include <rw/math/Pose6D.hpp>
#include <iostream>
#include <rtde.h>

class URControl {
    using Q = rw::math::Q;
private:
    ros::NodeHandle nh;
    rw::models::WorkCell::Ptr wc = rw::loaders::WorkCellLoader::Factory::load(ros::package::getPath("mergable_industrial_robots") + "/WorkCell/Scene.wc.xml");
    rw::models::Device::Ptr device = wc->findDevice("UR5");
    rw::kinematics::State state = wc->getDefaultState();
    rw::proximity::CollisionDetector::Ptr detector = new rw::proximity::CollisionDetector(wc, rwlibs::proximitystrategies::ProximityStrategyFactory::makeDefaultCollisionStrategy());
    ur_rtde::RTDEReceiveInterface *rtdeReceive = nullptr;
    ur_rtde::RTDEControlInterface *rtdeControl = nullptr;
    ur_rtde::RTDEIOInterface *rtdeIO = nullptr;

    double robotSpeed = 1;
    double robotAcceleration = 1;

    uint8_t gripperSpeed = 255;
    uint8_t gripperForce = 255;

    bool setToolDigitalOut(std::bitset<8>);
    bool initGripper();

public:

    explicit URControl(std::string);

    //Util
    std::vector<double> qToVector(Q q);
    std::vector<double> transformToVector(rw::math::Transform3D<double>);

    //UR Related
    bool moveHome();
    Q getQ();
    void setSpeed(double);
    void setAcceleration(double);
    bool checkCollision(Q q);     //Check if given configuration 'q' is in collision:
    bool move(Q q);
    bool move(Q q, double speed, double acceleration);
    bool moveToPose(rw::math::Transform3D<> T_BT_desired);
    bool moveRelative(Q dq);

    //Gripper related
    bool moveGripper(uint8_t);
    bool closeGripper();
    bool openGripper();


    Q randomConfig();
};


#endif // URROBOT_H
