#include "rw/rw.hpp"
#include "caros/serial_device_si_proxy.h"
#include "ros/package.h"
#include <iostream>
#include <fstream>
#include <rw/kinematics.hpp>
#include <rw/kinematics/State.hpp>
#include <rw/proximity.hpp>
#include <rw/math/Transform3D.hpp>
#include <rwlibs/algorithms/PointPairsRegistration.hpp>
#include <rw/math/EAA.hpp>
#include <rw/math/Quaternion.hpp>
#include <ar_track_alvar_msgs/AlvarMarkers.h>
#include <rw/invkin/JacobianIKSolver.hpp>
#include <rw/invkin/InvKinSolver.hpp>
#include <geometry_msgs/Transform.h>
#include <std_msgs/Float64.h>


//Collision detection
#include <rw/proximity/CollisionDetector.hpp>
#include <rw/proximity/ProximityStrategy.hpp>
#include <rwlibs/proximitystrategies/ProximityStrategyFactory.hpp>

//Includes for file reading
#include <fstream>
#include <sstream>
#include <string>


//FOR CALLBACK AND COMUNICATION
bool newDataAvailable = true;
rw::math::Transform3D<double> cbTransform;
//baseTcam Transform (Estimated by jimmi group)

rw::math::Rotation3D<double> baseTcamRos(   -0.0156, 0.9994, 0.0303,
                                            -0.5083, 0.0182, -0.8610,
                                            -0.8611, -0.0288, 0.5077);

rw::math::Vector3D<double> bateTcamDisp(0.3834,0.8302,0.6043);
rw::math::Transform3D<double> baseTcam(bateTcamDisp, baseTcamRos);

//AR-TRACKER
rw::math::Vector3D<> markerPointsTbase;
rw::math::Vector3D<> camPointsTmarker;
std::vector<rw::math::Vector3D<>> camPointsTmarkerVec;
std::vector<rw::math::Vector3D<>> markerPointsTbaseVec;
typedef std::pair< rw::math::Vector3D<>, rw::math::Vector3D<>> 	PointPair;
std::ofstream markerTransFile ("markerTransfile");
rw::math::Transform3D<> cameraTbase;
rw::math::Transform3D<> markerTcamera;
rw::math::Transform3D<> markerTbase;
std::vector<rw::math::Transform3D<>> cameraTbaseVec;
rw::math::Vector3D<> eaaComplete;
rw::math::Vector3D<> posComplete;

//Globals for Collisionchecking
double epsilon = 0.01;
int collisionCheck = 0;

class URRobot {
	using Q = rw::math::Q;


private:
	ros::NodeHandle nh;
	rw::models::WorkCell::Ptr wc;
	rw::models::Device::Ptr device;
	rw::kinematics::State state;
	caros::SerialDeviceSIProxy* robot;
    rw::proximity::CollisionDetector::Ptr detector;


public:
	URRobot()
	{
		auto packagePath = ros::package::getPath("rovi2");
		wc = rw::loaders::WorkCellLoader::Factory::load(packagePath + "/WorkCell/Scene.wc.xml");
		device = wc->findDevice("UR5");
		state = wc->getDefaultState();
		robot = new caros::SerialDeviceSIProxy(nh, "caros_universalrobot");
        detector = new rw::proximity::CollisionDetector(wc, rwlibs::proximitystrategies::ProximityStrategyFactory::makeDefaultCollisionStrategy());
		// Wait for first state message, to make sure robot is ready
		ros::topic::waitForMessage<caros_control_msgs::RobotState>("/caros_universalrobot/caros_serial_device_service_interface/robot_state", nh);
	    ros::spinOnce();
	}

	Q getQ()
	{
		// spinOnce processes one batch of messages, calling all the callbacks
	    ros::spinOnce();
	    Q q = robot->getQ();
		device->setQ(q, state);
	    return q;
	}

    bool setQ(Q q, caros::SerialDeviceSIProxy* dev)
    {
        // Tell robot to move to joint config q
        float speed = 0.2;
        if (dev->movePtp(q, speed)) {
            return true;
        } else
            return false;
    }


    bool setQ(Q q)
	{
		// Tell robot to move to joint config q
        float speed = 0.2;
		if (robot->movePtp(q, speed)) {
		    device->setQ(q,state);
			return true;
		} else
			return false;
	}

	bool moveQ(Q q){
        return setQ(robot->getQ() + q);
	}

	bool moveHome(){
        Q defaultQ = device->getQ(wc->getDefaultState());
        return setQ(defaultQ);
	}

	rw::math::Transform3D<> getPose(){
	    return device->getEnd()->getTransform(state);
	}


    rw::kinematics::State getState(){
	    return state;
	}

    rw::models::Device::Ptr getDevice(){
	    return device;
	}

	rw::math::Transform3D<double> getbaseTend(){
	    return getDevice()->baseTend(state);
	}

    std::vector<rw::math::Q> inverseKin(rw::math::Transform3D<> transform){
        rw::invkin::JacobianIKSolver invK(getDevice(),getState());
        invK.setMaxError(0.00001);
        invK.setMaxIterations(1000);
        return invK.solve(transform, getState());
    }


	//TAKEN FROM BasicCollision.cpp FROM collisionTest_Result ROVI2 BlackBoard
    bool inCollision(Q q){
        device->setQ(q, state);
        rw::proximity::CollisionDetector::QueryResult data;
        collisionCheck++;
        bool collision = detector->inCollision(state,&data);
        return collision;
    }

    bool binarySearch(Q q1, Q q2){
        Q dq = q2-q1;
        double N = dq.norm2()/epsilon;
        int levels = ceil(log2(N));
        Q qi;
        for (int i = 1; i <= levels; i++){
            int steps = pow(2, i-1);
            Q step = dq/steps;
            for (int k = 1; k <= steps; k++){
                qi = q1+((double)k-0.5)*step;
                if (inCollision(qi)){
                    return false;
                }
            }
        }
        return true;
    }

};

void transformCallback(geometry_msgs::TransformPtr transform){
    rw::math::Vector3D<double> translation = rw::math::Vector3D<double>(transform->translation.x,transform->translation.y,transform->translation.z);
    rw::math::Quaternion<double> rotationQuaternion = rw::math::Quaternion<double>(transform->rotation.x,transform->rotation.y,transform->rotation.z,transform->rotation.w);
    cbTransform = rw::math::Transform3D<double>(translation,rotationQuaternion);
    std::cout << "cbTransform: " << cbTransform << std::endl;
    newDataAvailable = 1;
}
void tester(){

    //INIT ROBOT
    URRobot robot;

    //Define Q's for caliration
    rw::math::Q q1(6, 0.7732124328613281, -1.0053818982890625, 2.140766445790426, -2.1643673382201136, 1.160172939300537, -1.7588465849505823);
    rw::math::Q q2(6, 0.5958089828491211, -0.5720837873271485, 1.3198588530169886, -2.1279221973814906, 0.7736911773681641, -2.633404795323507);
    rw::math::Q q3(6, 0.8070425987243652, -0.9067686361125489, 1.3945773283587855, -1.5328548711589356, 0.9157900810241699, -3.199463669453756);
    rw::math::Q q4(6, 0.7325644493103027, -0.9267538350871583, 1.7901080290423792, -2.11643185238027, 1.0915303230285645, -4.623188320790426);

    std::vector<rw::math::Q> qs{q1,q2,q3,q4};

    //Call this function to run the callibration
    //arTracker();

    rw::math::Q q(6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1);

    rw::math::Transform3D<double> trans;

    if(robot.moveHome()){
        std::cout << std::endl << "Successfully moved robot" << std::endl;
    }else{std::cout << std::endl << "Failed to move robot" << std::endl;}

    rw::math::Q initialQ = robot.getQ();

    if(robot.moveQ(q)){
        std::cout << std::endl << "Successfully moved robot" << std::endl;
    }else{std::cout << std::endl << "Failed to move robot" << std::endl;}


    trans = robot.getDevice()->baseTend(robot.getState());
    std::cout << "Transform" << trans << std::endl;

    std::vector<rw::math::Q> qInv = robot.inverseKin(trans);
    std::cout << "q Size: " << qInv.size() << std::endl;
    std::cout << "Initial q: " << initialQ << std::endl;
    std::cout << "Inv kin q: " << qInv[0] << std::endl;

    if(!robot.binarySearch(initialQ,qInv[0])){
        std::cout << "COLLISION DETECTION FAILED" << std::endl;
    }else{std::cout << "NO COLLISIONS DETECTED" << std::endl;}

    std::cout << "CollisionChecks: " << collisionCheck << std::endl;

    if(robot.moveHome()){
        std::cout << std::endl << "Successfully moved robot" << std::endl;
    }else{std::cout << std::endl << "Failed to move robot" << std::endl;}

    if(robot.setQ(qInv[0])){
        std::cout << std::endl << "Successfully moved robot" << std::endl;
    }else{std::cout << std::endl << "Failed to move robot" << std::endl;}

}


int main(int argc, char** argv)
{
	ros::init(argc, argv, "URRobot");
	ros::NodeHandle n("~");

    //INIT ROBOT
    URRobot robot;

    ros::Subscriber sub = n.subscribe("/rovi2/camTobject", 1, &transformCallback);

    //Gripper Publisher
    ros::Publisher pub = n.advertise<std_msgs::Float64>("rovi2/gripperMsg", 1);
    //readTransformFromFile(argv[1]);
    std_msgs::Float64 gripperConf;
    gripperConf.data = 0.03;


    rw::math::Vector3D<double> translation = rw::math::Vector3D<double>(-0.000799005, -0.284, 0.99963);
    rw::math::Rotation3D<double> rotation = rw::math::Rotation3D<double>(3.64677e-15, 0.999999, -0.00159255, -1, 3.64666e-15, -7.01612e-17, -6.43537e-17, 0.00159255, 0.999999);
    rw::math::Transform3D<double> trans(translation,rotation);
    ros::Rate loop_rate(10);
    while(ros::ok()) {
        if(newDataAvailable){
            if(robot.moveHome()){
                std::cout << std::endl << "Successfully moved robot" << std::endl;
            }else{std::cout << std::endl << "Failed to move robot" << std::endl;}

            newDataAvailable = false;
        }
    ros::spinOnce();
    loop_rate.sleep();
    }

    ros::spinOnce();

    return 0;
}
