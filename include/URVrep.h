//
// Created by kasper on 9/23/19.
//

#ifndef MERGABLEINDUSTRIALROBOTS_URVREPSIM_H
#define MERGABLEINDUSTRIALROBOTS_URVREPSIM_H

#include <iostream>

#include <ros/node_handle.h>
#include <ros/package.h>
#include <ros/publisher.h>
#include <rw/math/Q.hpp>
#include <rw/models/Device.hpp>
#include <rw/kinematics/State.hpp>
#include <rw/models/WorkCell.hpp>
#include <rw/proximity/CollisionDetector.hpp>
#include <rw/loaders/WorkCellLoader.hpp>
#include <std_msgs/Float32MultiArray.h>
//Collision detection
#include <rwlibs/proximitystrategies/ProximityStrategyFactory.hpp>
#include "mergable_industrial_robots/moveRobot.h"
#include <chrono>
#include <thread>
#include <std_msgs/Bool.h>

class URVrep {
    using Q = rw::math::Q;


private:
    bool callVrepService(Q);
    ros::ServiceClient client;

    ros::Publisher startSimPublisher;
    ros::Publisher stopSimPublisher;

    //RW and collision detection
    ros::NodeHandle nh;
    rw::models::WorkCell::Ptr wc = rw::loaders::WorkCellLoader::Factory::load(ros::package::getPath("mergable_industrial_robots") + "/WorkCell/Scene.wc.xml");
    rw::models::Device::Ptr device = wc->findDevice("UR5");
    rw::kinematics::State state = wc->getDefaultState();
    rw::proximity::CollisionDetector::Ptr detector = new rw::proximity::CollisionDetector(wc, rwlibs::proximitystrategies::ProximityStrategyFactory::makeDefaultCollisionStrategy());
    Q defaultQ = device.get()->getQ(state);
public:
    URVrep(const std::string&);

    Q getQ();

    bool setQ(Q);
    bool moveQ(Q);
    bool moveHome();
    rw::kinematics::State getState();
    rw::models::Device::Ptr getDevice();
    void publishQ(Q, ros::Publisher);
    void setServiceName(std::string);
    void startSim();
    void stopSim();
};


#endif //MERGABLEINDUSTRIALROBOTS_URVREPSIM_H
