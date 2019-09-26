#include "../include/URrobot.h"


URRobot::URRobot()
{
    auto packagePath = ros::package::getPath("mergable_industrial_robots");
    wc = rw::loaders::WorkCellLoader::Factory::load(packagePath + "/WorkCell/Scene.wc.xml");
    device = wc->findDevice("UR5");
    state = wc->getDefaultState();
    robot = new caros::SerialDeviceSIProxy(nh, "caros_universalrobot");

    //shp:
    detector = new CollisionDetector(wc, ProximityStrategyFactory::makeDefaultCollisionStrategy());

    // Wait for first state message, to make sure robot is ready
    ros::topic::waitForMessage<caros_control_msgs::RobotState>("/caros_universalrobot/caros_serial_device_service_interface/robot_state", nh);
    ros::spinOnce();
}

Q URRobot::getQ()
{
    // spinOnce processes one batch of messages, calling all the callbacks
    ros::spinOnce();
    Q q = robot->getQ();
    device->setQ(q, state);
    return q;
}

bool URRobot::setQ(Q q)
{
    // Tell robot to move to joint config q
    float speed = 0.5;
    if (robot->movePtp(q, speed)) {
        return true;
    } else
        return false;
}

bool URRobot::moveQ(Q dq)
{
    Q currentQ = getQ();
    Q desiredQ = currentQ + dq;
    return setQ(desiredQ);
}

bool URRobot::goToZeroPos()
{
    State default_state = wc->getDefaultState();
    Q homeQ = device->getQ(default_state);
    cout << "HomeQ: " << homeQ << endl;
    return setQ(homeQ);
}

bool URRobot::checkCollision(Q q)
{
    device->setQ(q, state);
    CollisionDetector::QueryResult data;
    bool collision = detector->inCollision(state,&data);

    getQ(); //reset q to actual config:

    return collision;
}

bool URRobot::moveToPose(Transform3D<> T_BT_desired)
{
    robot->movePtpT(T_BT_desired);
}

bool URRobot::moveToPose(RPY<> RPY_desired, Vector3D<> displacement_desired)
{
    Transform3D<> T_BT_desired(displacement_desired, RPY_desired.toRotation3D());
    robot->movePtpT(T_BT_desired);
}

bool URRobot::moveToPose(double arr[]) //State-vector: [X,Y,Z,R,P,Y]
{
    Vector3D<> disp_des(arr[0], arr[1], arr[2]);
    RPY<> RPY_des(arr[3], arr[4], arr[5]);
    Transform3D<> T_BT_desired(disp_des,RPY_des.toRotation3D());
    robot->movePtpT(T_BT_desired);
}

Transform3D<> URRobot::getPose()
{
    //List every frame:
    for(auto frame : wc->getFrames())
        cout << frame->getName() << endl;

    //Transform3D<> T_BT = device->baseTend(state);
    cout << "--------- Base to TCP ---------" << endl;
    Frame* TCP_Frame = wc->findFrame("UR5.TCP");
    Transform3D<> T_BT = device->baseTframe(TCP_Frame, state);
    Vector3D<> d_BT = T_BT.P();      //Displacement from base
    Rotation3D<> R_BT = T_BT.R(); //Extract rotation matrix
    RPY<> RPY_obj(R_BT);
    cout << T_BT.e() << endl;

    cout << "RPY of current T_BT" << endl;
    cout << RPY_obj << endl;

    cout << "--------- Base to Toolbase ---------" << endl;
    Frame* Toolbase_Frame = wc->findFrame("UR5.ToolBase");
    Transform3D<> T_BToolbase = device->baseTframe(Toolbase_Frame, state);
    Vector3D<> d_BToolbase = T_BToolbase.P();      //Displacement from base
    Rotation3D<> R_BToolbase = T_BToolbase.R(); //Extract rotation matrix
    RPY<> RPY_Toolbase(R_BT);
    cout << T_BToolbase.e() << endl;
    cout << "RPY of T_BToolbase" << endl;
    cout << RPY_Toolbase << endl;

    //robot->

    return T_BT;
}

Eigen::Matrix<double,6,1> URRobot::computeTaskError(Q qnear, Q qs)
{
    Frame* taskFrame = wc->findFrame("TaskFrame");
    Transform3D<> TBaseTask = device->baseTframe(taskFrame, state);

    cout << "Transform base to task" << endl;
    cout << TBaseTask.e() << endl;

    Frame* endFrame = wc->findFrame("UR5.ToolBase");
    Transform3D<> TTaskEnd = taskFrame->fTf(endFrame, state);
    cout << "TTaskEnd: " << endl;
    cout << TTaskEnd.e() << endl;

    Transform3D<> TBaseEnd = device->baseTend(state);

    Rotation3D<> R_diff = TTaskEnd.R();
    RPY<> RPY_diff(R_diff);
    Vector3D<> disp_diff = TTaskEnd.P();

    Eigen::Matrix<double,6,1> dx;
    dx << disp_diff[0], disp_diff[1], disp_diff[2], RPY_diff[2], RPY_diff[1], RPY_diff[0]; //[XYZ YPR]
    Eigen::Matrix<double,6,6> C;
    C <<    0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1;

    cout << "dx: \n" << dx;
    cout << "C*dx: \n" << C * dx << endl;
    dx = C * dx;
    return dx;

}

Q URRobot::randomConfig()
{
    Q qMin = device->getBounds().first;
    Q qMax = device->getBounds().second;
    Math::seed();
    Q q1 = Math::ranQ(qMin,qMax);
    return q1;
}


void URRobot::follow_path(vector<Q> path)
{
    for(int i = 0; i < path.size(); i++ )
        setQ(path[i]);
}


int myfunc(int a, int b)
{
    return a + b;
}
