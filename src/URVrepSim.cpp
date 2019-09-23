//
// Created by kasper on 9/23/19.
//

#include "../include/URVrepSim.h"


URVrepSim::URVrepSim() {
    //auto packagePath = ros::package::getPath("");
    //wc = rw::loaders::WorkCellLoader::Factory::load(packagePath + "/WorkCell/Scene.wc.xml");
    //device = wc->findDevice("UR5");
    //state = wc->getDefaultState();
    //detector = new rw::proximity::CollisionDetector(wc, rwlibs::proximitystrategies::ProximityStrategyFactory::makeDefaultCollisionStrategy());
}

URVrepSim::Q URVrepSim::getQ() {
    return URVrepSim::Q();
}

bool URVrepSim::setQ(URVrepSim::Q) {
    return true;
}

bool URVrepSim::moveQ(URVrepSim::Q) {
    return true;
}

bool URVrepSim::moveHome() {
    return true;
}

rw::kinematics::State URVrepSim::getState() {
    return state;
}

rw::models::Device::Ptr URVrepSim::getDevice() {
    return device;
}

//Execute what the node should do underneath
int main(){
    std::cout << "Hello World!" << std::endl;



}