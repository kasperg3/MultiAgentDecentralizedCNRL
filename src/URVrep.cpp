//
// Created by kasper on 9/23/19.
//

#include "URVrep.h"

URVrep::URVrep() {
    auto packagePath = ros::package::getPath("mergable_industrial_robots");
    wc = rw::loaders::WorkCellLoader::Factory::load(packagePath + "/WorkCell/Scene.wc.xml");
    device = wc->findDevice("UR5");
    state = wc->getDefaultState();
    detector = new rw::proximity::CollisionDetector(wc, rwlibs::proximitystrategies::ProximityStrategyFactory::makeDefaultCollisionStrategy());
    defaultQ = this->getDevice().get()->getQ(state);
    ros::NodeHandle nh;
    //Set the default Q
    defaultQ = this->getDevice().get()->getQ(state);
    //Create publishers for starting and stopping the sim
    stopSimPublisher = nh.advertise<std_msgs::Bool>("/stopSimulation", 1);
    startSimPublisher = nh.advertise<std_msgs::Bool>("/startSimulation", 1);
}

////TODO: IMPLEMENT THIS
URVrep::Q URVrep::getQ() {
    return URVrep::Q();
}

bool URVrep::setQ(URVrep::Q q) {
    this->device.get()->setQ(q, state );
    //init clock for timeput
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    //Stay in while loop until timeout or call is complete
    while(!callVrepService(q)) {
        if(std::chrono::steady_clock::now() - start > std::chrono::seconds(20))
            break;
        std::chrono::milliseconds timespan(100); // or whatever
        std::this_thread::sleep_for(timespan);
    }
    return true;
}

bool URVrep::moveQ(URVrep::Q q) {
    Q currentQ = this->device->getQ(state);
    return setQ(currentQ + q);
}

bool URVrep::moveHome() {
    return setQ(defaultQ);
}

rw::kinematics::State URVrep::getState() {
    return state;
}

rw::models::Device::Ptr URVrep::getDevice() {
    return device;
}

void URVrep::publishQ(URVrep::Q q, ros::Publisher pub) {
    std_msgs::Float32MultiArray msg;
    msg.data.clear();
    for(unsigned int i = 0; i < q.size(); i++){
            msg.data.push_back((float)q(i) - defaultQ(i));
    }
    pub.publish(msg);
}

bool URVrep::callVrepService(URVrep::Q q) {
    mergable_industrial_robots::moveRobot srv;

    for (unsigned int i = 0; i < q.size(); i++) {
        srv.request.requestQ.push_back((float) q(i)- defaultQ(i));
    }
    if (client.call(srv.request,srv.response)) {
    } else {
        std::string error("Failed to call service: ");
        ROS_ERROR((const char*)error.append(client.getService()).c_str());
    }
    if (srv.response.response == true)
        return true;
    else
        return false;
}

void URVrep::setServiceName(std::string serviceName) {
    client = nh.serviceClient<mergable_industrial_robots::moveRobot>(serviceName);
}

void URVrep::startSim() {
    startSimPublisher.publish(std_msgs::Bool());
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    ros::spinOnce();
}

void URVrep::stopSim() {
    stopSimPublisher.publish(std_msgs::Bool());
    ros::spinOnce();
}

