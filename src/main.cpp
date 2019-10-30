//
// Created by kasper on 9/27/19.
//

#include <ros/ros.h>
#include "mergable_industrial_robots/moveRobot.h"
#include <rw/math/Q.hpp>
#include "URVrepSim.h"
#include "TTTRL.h"

void testVrep(){
    ros::Rate loop_rate(100);

//Define Q's
    rw::math::Q q1(6, 0.7732124328613281, -1.0053818982890625, 2.140766445790426, -2.1643673382201136,
                   1.160172939300537, -1.7588465849505823);
    rw::math::Q q2(6, 0.5958089828491211, -0.5720837873271485, 1.3198588530169886, -2.1279221973814906,
                   0.7736911773681641, -2.633404795323507);
    rw::math::Q q3(6, 0.8070425987243652, -0.9067686361125489, 1.3945773283587855, -1.5328548711589356,
                   0.9157900810241699, -3.199463669453756);
    rw::math::Q q4(6, 0.7325644493103027, -0.9267538350871583, 1.7901080290423792, -2.11643185238027,
                   1.0915303230285645, -4.623188320790426);

    rw::math::Q qtest(6, 0, 0, 0, 0, 0, 0);
    //  qtest = Q(6, -0.979, -0.859, 1.195, 1.236, 1.571, -2.55); //beautiful bottlegrab -40 40 10 0 0 180
    //  qtest = Q(6, 2.547, -2.14, -1.939, -0.639, 1.57, 0.977);
    qtest = rw::math::Q(6, 0, -2.14, -1.939, -0.639, 1.57, 0.977); //beautiful box grab

    URVrepSim robot0;
    URVrepSim robot1;

    robot0.setServiceName("/vrep_ros_interface/moveRobot0");
    robot1.setServiceName("/vrep_ros_interface/moveRobot1");
    int dummy = 0;
    while (ros::ok()) {
        if(dummy % 2){
            robot0.moveHome();
            robot1.moveHome();
        }else{
            robot0.setQ(qtest);
            robot1.setQ(qtest);
        }
        dummy++;
        ros::spinOnce();
        loop_rate.sleep();
    }
}

//Not working
//void testURSim(){
//    //INIT ROBOT
//    ros::Rate loop_rate(10);
//    URRobot robot;
//    while(ros::ok()) {
//        if(robot.moveHome()){
//            std::cout << std::endl << "Successfully moved robot" << std::endl;
//        }else{std::cout << std::endl << "Failed to move robot" << std::endl;}
//
//        ros::spinOnce();
//        loop_rate.sleep();
//    }
//
//
//}

int main(int argc, char **argv) {
    ros::init(argc, argv, "URVrepSim");
    ros::NodeHandle n("~");

    //testVrep();

    TTTRL game;




}