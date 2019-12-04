#include "URControl.h"

URControl::URControl(std::string robot_ip){
    ROS_INFO("[RobotController] Connecting to Control interface");
    rtdeControl = new ur_rtde::RTDEControlInterface(robot_ip);
    while(!rtdeControl->isConnected()){
        rtdeControl->reconnect();               //try reconnect to robot
    }
    ROS_INFO("[RobotController] Connecting to Reciever interface");
    rtdeReceive = new ur_rtde::RTDEReceiveInterface(robot_ip);
    while(!rtdeReceive->isConnected()){
        rtdeReceive->reconnect();               //try reconnect to robot
    }
    ROS_INFO("[RobotController] Connecting to IO");
    rtdeIO = new ur_rtde::RTDEIOInterface(robot_ip);
    rtdeControl->servoStop();
    ROS_INFO("[ URRobot: ]");
}

URControl::Q URControl::getQ(){
    // spinOnce processes one batch of messages, calling all the callbacks
    ros::spinOnce();
    Q q = rtdeReceive->getActualQ();
    device->setQ(q, state);
    return q;
}

std::vector<double> URControl::qToVector(Q q){
    std::vector<double> tempVec = {};
    for(int i = 0; i < q.m().size(); i++){
        tempVec.push_back(q[i]);
    }
    return tempVec;
}

std::vector<double> URControl::transformToVector(rw::math::Transform3D<double> transform){
    rw::math::RPY rpy(transform.R());
    rw::math::Vector3D pos(transform.P());
    std::vector vec = {pos(0), pos(1),pos(3), rpy(0), rpy(1), rpy(2)};
    return vec;
}

bool URControl::moveRelative(Q dq){
    Q currentQ = getQ();
    Q desiredQ = currentQ + dq;
    return rtdeControl->moveJ(qToVector(desiredQ));
}

bool URControl::moveHome(){
    rw::kinematics::State default_state = wc->getDefaultState();
    Q homeQ = device->getQ(default_state);
    return rtdeControl->moveJ(qToVector(homeQ), robotSpeed,robotAcceleration);
}

bool URControl::move(Q q){
    return rtdeControl->moveJ(qToVector(q), robotSpeed, robotAcceleration);
}

bool URControl::moveToPose(rw::math::Transform3D<> T_BT_desired){
    return rtdeControl->moveJ_IK(transformToVector(T_BT_desired));
}

bool URControl::checkCollision(Q q){
    device->setQ(q, state);
    rw::proximity::CollisionDetector::QueryResult data;
    bool collision = detector->inCollision(state,&data);
    getQ(); //reset q to actual config:
    return collision;
}

URControl::Q URControl::randomConfig(){
    Q qMin = device->getBounds().first;
    Q qMax = device->getBounds().second;
    rw::math::Math::seed();
    Q q1 = rw::math::Math::ranQ(qMin,qMax);
    return q1;
}

void URControl::setSpeed(double speed) {
    robotSpeed = speed;
}

void URControl::setAcceleration(double acc) {
    robotAcceleration = acc;
}

bool URControl::move(URControl::Q q, double speed, double acceleration) {
    return rtdeControl->moveJ(qToVector(q), speed, acceleration);
}

bool URControl::moveGripper(uint8_t) {

    return false;
}

bool URControl::closeGripper() {
    return rtdeIO->setToolDigitalOut(0, false);
}

bool URControl::openGripper() {
    return rtdeIO->setToolDigitalOut(0, true);
}

bool URControl::setToolDigitalOut(std::bitset<8>) {

}
