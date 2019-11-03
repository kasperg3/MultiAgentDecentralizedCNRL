#include "URControl.h"

URControl::URControl(std::string robot_ip){
    ROS_INFO("robotController.cpp : Connecting to controller to the robot");
    rtdeControl = new ur_rtde::RTDEControlInterface(robot_ip);
    while(!rtdeControl->isConnected()){
        rtdeControl->reconnect();               //try reconnect to robot
    }
    ROS_INFO("robotController.cpp : Connecting receiver to the robot");
    rtdeReceive = new ur_rtde::RTDEReceiveInterface(robot_ip);
    while(!rtdeReceive->isConnected()){
        rtdeReceive->reconnect();               //try reconnect to robot
    }
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
    return rtdeControl->moveJ(qToVector(homeQ), 1,1);
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


rw::math::Transform3D<> URControl::getPose(){
    //List every frame:
    for(auto frame : wc->getFrames())
        std::cout << frame->getName() << std::endl;

    //Transform3D<> T_BT = device->baseTend(state);
    std::cout << "--------- Base to TCP ---------" << std::endl;
    rw::kinematics::Frame* TCP_Frame = wc->findFrame("UR5.TCP");
    rw::math::Transform3D<> T_BT = device->baseTframe(TCP_Frame, state);
    rw::math::Vector3D<> d_BT = T_BT.P();      //Displacement from base
    rw::math::Rotation3D<> R_BT = T_BT.R(); //Extract rotation matrix
    rw::math::RPY<> RPY_obj(R_BT);
    std::cout << T_BT.e() << std::endl;

    std::cout << "RPY of current T_BT" << std::endl;
    std::cout << RPY_obj << std::endl;

    std::cout << "--------- Base to Toolbase ---------" << std::endl;
    rw::kinematics::Frame* Toolbase_Frame = wc->findFrame("UR5.ToolBase");
    rw::math::Transform3D<> T_BToolbase = device->baseTframe(Toolbase_Frame, state);
    rw::math::Vector3D<> d_BToolbase = T_BToolbase.P();      //Displacement from base
    rw::math::Rotation3D<> R_BToolbase = T_BToolbase.R(); //Extract rotation matrix
    rw::math::RPY<> RPY_Toolbase(R_BT);
    std::cout << T_BToolbase.e() << std::endl;
    std::cout << "RPY of T_BToolbase" << std::endl;
    std::cout << RPY_Toolbase << std::endl;

    return T_BT;
}

Eigen::Matrix<double,6,1> URControl::computeTaskError(Q qnear, Q qs){
    rw::kinematics::Frame* taskFrame = wc->findFrame("TaskFrame");
    rw::math::Transform3D<> TBaseTask = device->baseTframe(taskFrame, state);

    std::cout << "Transform base to task" << std::endl;
    std::cout << TBaseTask.e() << std::endl;

    rw::kinematics::Frame* endFrame = wc->findFrame("UR5.ToolBase");
    rw::math::Transform3D<> TTaskEnd = taskFrame->fTf(endFrame, state);
    std::cout << "TTaskEnd: " << std::endl;
    std::cout << TTaskEnd.e() << std::endl;

    rw::math::Transform3D<double> TBaseEnd = device->baseTend(state);

    rw::math::Rotation3D<> R_diff = TTaskEnd.R();
    rw::math::RPY<> RPY_diff(R_diff);
    rw::math::Vector3D<> disp_diff = TTaskEnd.P();

    Eigen::Matrix<double,6,1> dx;
    dx << disp_diff[0], disp_diff[1], disp_diff[2], RPY_diff[2], RPY_diff[1], RPY_diff[0]; //[XYZ YPR]
    Eigen::Matrix<double,6,6> C;
    C <<    0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1;

    std::cout << "dx: \n" << dx;
    std::cout << "C*dx: \n" << C * dx << std::endl;
    dx = C * dx;
    return dx;

}

URControl::Q URControl::randomConfig(){
    Q qMin = device->getBounds().first;
    Q qMax = device->getBounds().second;
    rw::math::Math::seed();
    Q q1 = rw::math::Math::ranQ(qMin,qMax);
    return q1;
}