//
// Created by kasper on 9/23/19.
//

#include "URVrep.h"

/*
 * ROBOT Number can be wither 0 or 1, others are not completed
 */
URVrep::URVrep(const std::string& robotNumber) {

    if(robotNumber != "1" && robotNumber != "0"){
        ROS_ERROR("[URVrep] NOT A VALID ROBOT NUMBER");
        throw "[URVrep] NOT A VALID ROBOT NUMBER";
    }
    //Create publishers for starting and stopping the sim
    stopSimPublisher = nCtrl.advertise<std_msgs::Bool>("/stopSimulation", 5);
    startSimPublisher = nCtrl.advertise<std_msgs::Bool>("/startSimulation", 5);
    //Create subscriber for simulation state:
    simStateSubscriber = nCtrl.subscribe("/simulationState", 2, &URVrep::stateCallback, this);
    //Setup gripper control
    gripperSimPublisher = nCtrl.advertise<std_msgs::Bool>("/closeGripper" + robotNumber, 5);
    //Moving subscriber
    isMovingSubscriber = nCtrl.subscribe("/robotMoving" + robotNumber, 2, &URVrep::robotMovingCallback, this);
    //Robot Q publisher
    robotQPublisher = nCtrl.advertise<std_msgs::Float32MultiArray>("/robotQ" + robotNumber, 5);
}

URVrep::~URVrep() {
    stopSim();
    stopSimPublisher.shutdown();
    startSimPublisher.shutdown();
    simStateSubscriber.shutdown();
    robotQPublisher.shutdown();
    isMovingSubscriber.shutdown();
}

void URVrep::stateCallback(const std_msgs::Int32::ConstPtr& msg){
    simState = msg.get()->data;
}

void URVrep::robotMovingCallback(const std_msgs::Bool::ConstPtr &msg) {
    isMoving = msg.get()->data;
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

    while(isRobotMoving()) {   //Wait until done moving or timeout
        if(std::chrono::steady_clock::now() - start > std::chrono::seconds(20)) {
            return false;
        }
        std::chrono::milliseconds timespan(100); // or whatever
        std::this_thread::sleep_for(timespan);
    }

    publishQ(q, robotQPublisher);
    ros::spinOnce();

    //Sleep to not spam the robot with move requests until is has started moving
    std::chrono::milliseconds timespan(500); // or whatever
    std::this_thread::sleep_for(timespan);
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

void URVrep::closeGripper() {
    std_msgs::Bool msg;
    msg.data = true;
    gripperSimPublisher.publish(msg);
    ros::spinOnce();

}

void URVrep::openGripper() {
    std_msgs::Bool msg;
    msg.data = false;
    gripperSimPublisher.publish(msg);
    ros::spinOnce();

}

bool URVrep::isRobotMoving() {
    ros::spinOnce();
    return isMoving;
}
