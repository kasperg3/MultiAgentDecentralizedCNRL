//
// Created by kasper on 9/23/19.
//

#include "URVrep.h"

URVrep::URVrep(const std::string& serviceName) {
    //Create publishers for starting and stopping the sim
    stopSimPublisher = nCtrl.advertise<std_msgs::Bool>("/stopSimulation", 5);
    startSimPublisher = nCtrl.advertise<std_msgs::Bool>("/startSimulation", 5);
    //Create subscriber for simulation state:
    simStateSubscriber = nCtrl.subscribe("/simulationState", 2, &URVrep::stateCallback, this);
    //Create service client
    client = nh.serviceClient<mergable_industrial_robots::moveRobot>(serviceName);
}

URVrep::~URVrep() {
    stopSim();
    stopSimPublisher.shutdown();
    startSimPublisher.shutdown();
    simStateSubscriber.shutdown();
    client.shutdown();
}

void URVrep::stateCallback(const std_msgs::Int32::ConstPtr& msg){
    simState = msg.get()->data;
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

rw::models::Device::Ptr URVrep::getDevice() {
    return device;
}

rw::kinematics::State URVrep::getState() {
    return state;
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

bool URVrep::simStopped(){
    if(simState == 0)
        return true;
    else
        return false;
}

bool URVrep::simRunning(){
    if(simState == 1)
        return true;
    else
        return false;
}


void URVrep::startSim() {
    ROS_INFO("Starting VRep simulation...");
    while(!simRunning()){
        startSimPublisher.publish(std_msgs::Bool());
        ros::spinOnce();
    }
    ROS_INFO("VRep simulation started");
}

void URVrep::stopSim() {
    ROS_INFO("Stopping VRep simulation...");
    while(!simStopped()){
        stopSimPublisher.publish(std_msgs::Bool());
        ros::spinOnce();
    }
    ROS_INFO("VRep simulation Stopped");
}
