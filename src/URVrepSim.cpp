//
// Created by kasper on 9/23/19.
//

#include "../include/URVrepSim.h"

URVrepSim::URVrepSim() {
    auto packagePath = ros::package::getPath("mergable_industrial_robots");
    wc = rw::loaders::WorkCellLoader::Factory::load(packagePath + "/WorkCell/Scene.wc.xml");
    device = wc->findDevice("UR5");
    state = wc->getDefaultState();
    detector = new rw::proximity::CollisionDetector(wc, rwlibs::proximitystrategies::ProximityStrategyFactory::makeDefaultCollisionStrategy());
    defaultQ = this->getDevice().get()->getQ(state);
    ros::NodeHandle nh;
    //Create Service for controlling the robot in V-rep
    client = nh.serviceClient<mergable_industrial_robots::moveRobot>("vrep_ros_interface/moveRobot");
    defaultQ = this->getDevice().get()->getQ(state);
}

URVrepSim::Q URVrepSim::getQ() {
    return URVrepSim::Q();
}

bool URVrepSim::setQ(URVrepSim::Q q) {
    this->device.get()->setQ(q, state );
    //init clock for timeput
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    //Stay in while loop until timeout or call is complete
    while(!callVrepService(q)) {
        if(std::chrono::steady_clock::now() - start > std::chrono::seconds(10))
            break;
        std::chrono::milliseconds timespan(100); // or whatever
        std::this_thread::sleep_for(timespan);
    }
    return true;
}

bool URVrepSim::moveQ(URVrepSim::Q q) {
    Q currentQ = this->device->getQ(state);
    return setQ(currentQ + q);
}

bool URVrepSim::moveHome() {
    return setQ(defaultQ);
}

rw::kinematics::State URVrepSim::getState() {
    return state;
}

rw::models::Device::Ptr URVrepSim::getDevice() {
    return device;
}

void URVrepSim::publishQ(URVrepSim::Q q, ros::Publisher pub) {
    std_msgs::Float32MultiArray msg;
    msg.data.clear();
    for(unsigned int i = 0; i < q.size(); i++){
            msg.data.push_back((float)q(i) - defaultQ(i));
    }
    pub.publish(msg);
}

bool URVrepSim::callVrepService(URVrepSim::Q q) {
    mergable_industrial_robots::moveRobot srv;

    for (unsigned int i = 0; i < q.size(); i++) {
        srv.request.requestQ.push_back((float) q(i)- defaultQ(i));
    }
    if (client.call(srv)) {
    } else {
        ROS_ERROR("Failed to call service");
        return false;
    }
    return true;
}

void URVrepSim::setServiceName(std::string serviceName) {
    client = nh.serviceClient<mergable_industrial_robots::moveRobot>(serviceName);
}

