//
// Created by kasper on 9/27/19.
//

#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <rw/math/Q.hpp>
#include "URVrep.h"
#include "URControl.h"
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
    rw::math::Q qtest2(6, -0.979, -0.859, 1.195, 1.236, 1.571, -2.55); //beautiful bottlegrab -40 40 10 0 0 180
    //  qtest = Q(6, 2.547, -2.14, -1.939, -0.639, 1.57, 0.977);
    qtest = rw::math::Q(6, 0, -2.14, -1.939, -0.639, 1.57, 0.977); //beautiful box grab

    URVrep robot0 = URVrep("/vrep_ros_interface/moveRobot0");
    URVrep robot1 = URVrep("/vrep_ros_interface/moveRobot1");
    robot0.startSim();
    //robot0.stopSim();


    ros::NodeHandle n;
    ros::Publisher suctionPad1chatter = n.advertise<std_msgs::Bool>("/suctionPad1",1000);
   // b0RemoteApi client("b0RemoteApi_c++Client","b0RemoteApi");
   // cl=&client;
   // std::vector<bool> response = client.simxSetBoolParameter();
   // std::cout << response[1] << std::endl;
    robot0.startSim();
    int dummy = 0;
    std_msgs::Bool msg1;
    while (ros::ok()) {
        if(dummy % 3 == 0){
            msg1.data = false;
            suctionPad1chatter.publish(msg1);
            robot0.moveHome();
            std::cout << "moveHome" << std::endl;
            robot1.moveHome();

        }else if (dummy % 3 == 1){
            msg1.data = false;
            suctionPad1chatter.publish(msg1);
            robot0.setQ(qtest);
            std::cout << "setQ" << std::endl;
            robot1.setQ(qtest);

        }else{
            msg1.data = true;
            suctionPad1chatter.publish(msg1);
            robot0.setQ(qtest2);
            std::cout << "setQ2" << std::endl;
            robot1.setQ(qtest2);

        }
        dummy++;
        std::cout << "dummy: " << dummy << std::endl;

        ros::spinOnce();
        loop_rate.sleep();
    }
}

void testURControl(std::string ip){
    URControl robot(ip);
    robot.moveHome();
}

void TTTGame(){
    TTTRL game;
    game.playGames(400000);
    //game.playGames(10000);
    //game.playGame();

}

int main(int argc, char **argv) {
    ros::init(argc, argv, "URVrepSim");
    ros::NodeHandle n("~");

    testVrep();
    //testURControl("127.0.0.1");
    //TTTGame();
}