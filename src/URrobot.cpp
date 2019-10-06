#include "../include/URrobot.h"


URRobot::URRobot(){
    auto packagePath = ros::package::getPath("mergable_industrial_robots");
    wc = rw::loaders::WorkCellLoader::Factory::load(packagePath + "/WorkCell/Scene.wc.xml");
    device = wc->findDevice("UR5");
    state = wc->getDefaultState();
    robot = new caros::SerialDeviceSIProxy(nh, "caros_universalrobot");

    //shp:
    detector = new rw::proximity::CollisionDetector(wc, rwlibs::proximitystrategies::ProximityStrategyFactory::makeDefaultCollisionStrategy());

    // Wait for first state message, to make sure robot is ready
    ros::topic::waitForMessage<caros_control_msgs::RobotState>("/caros_universalrobot/caros_serial_device_service_interface/robot_state", nh);
    ros::spinOnce();
}

URRobot::Q URRobot::getQ(){
    // spinOnce processes one batch of messages, calling all the callbacks
    ros::spinOnce();
    Q q = robot->getQ();
    device->setQ(q, state);
    return q;
}

bool URRobot::setQ(Q q){
    // Tell robot to move to joint config q
    float speed = 0.5;
    if (robot->movePtp(q, speed)) {
        return true;
    } else
        return false;
}

bool URRobot::moveQ(Q dq){
    Q currentQ = getQ();
    Q desiredQ = currentQ + dq;
    return setQ(desiredQ);
}

bool URRobot::moveHome(){
    rw::kinematics::State default_state = wc->getDefaultState();
    Q homeQ = device->getQ(default_state);
    return setQ(homeQ);
}

bool URRobot::checkCollision(Q q){
    device->setQ(q, state);
    rw::proximity::CollisionDetector::QueryResult data;
    bool collision = detector->inCollision(state,&data);

    getQ(); //reset q to actual config:

    return collision;
}

bool URRobot::moveToPose(rw::math::Transform3D<> T_BT_desired){
    robot->movePtpT(T_BT_desired);
}

bool URRobot::moveToPose(rw::math::RPY<> RPY_desired, rw::math::Vector3D<> displacement_desired){
    rw::math::Transform3D<> T_BT_desired(displacement_desired, RPY_desired.toRotation3D());
    robot->movePtpT(T_BT_desired);
}

bool URRobot::moveToPose(double arr[]){ //State-vector: [X,Y,Z,R,P,Y]
    rw::math::Vector3D<> disp_des(arr[0], arr[1], arr[2]);
    rw::math::RPY<> RPY_des(arr[3], arr[4], arr[5]);
    rw::math::Transform3D<> T_BT_desired(disp_des,RPY_des.toRotation3D());
    robot->movePtpT(T_BT_desired);
}

rw::math::Transform3D<> URRobot::getPose(){
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

    //robot->

    return T_BT;
}

Eigen::Matrix<double,6,1> URRobot::computeTaskError(Q qnear, Q qs){
    rw::kinematics::Frame* taskFrame = wc->findFrame("TaskFrame");
    rw::math::Transform3D<> TBaseTask = device->baseTframe(taskFrame, state);

    std::cout << "Transform base to task" << std::endl;
    std::cout << TBaseTask.e() << std::endl;

    rw::kinematics::Frame* endFrame = wc->findFrame("UR5.ToolBase");
    rw::math::Transform3D<> TTaskEnd = taskFrame->fTf(endFrame, state);
    std::cout << "TTaskEnd: " << std::endl;
    std::cout << TTaskEnd.e() << std::endl;

    rw::math::Transform3D<> TBaseEnd = device->baseTend(state);

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

URRobot::Q URRobot::randomConfig(){
    Q qMin = device->getBounds().first;
    Q qMax = device->getBounds().second;
    rw::math::Math::seed();
    Q q1 = rw::math::Math::ranQ(qMin,qMax);
    return q1;
}


void URRobot::follow_path(std::vector<Q> path){
    for(int i = 0; i < path.size(); i++ )
        setQ(path[i]);
}