//
// Created by kasper on 9/23/19.
//

#include "../include/URVrepSim.h"


URVrepSim::URVrepSim() {
    auto packagePath = ros::package::getPath("");
    wc = rw::loaders::WorkCellLoader::Factory::load(packagePath + "/WorkCell/Scene.wc.xml");
    device = wc->findDevice("UR5");
    state = wc->getDefaultState();
    detector = new rw::proximity::CollisionDetector(wc, rwlibs::proximitystrategies::ProximityStrategyFactory::makeDefaultCollisionStrategy());
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

//Define Q's
    rw::math::Q q1(6, 0.7732124328613281, -1.0053818982890625, 2.140766445790426, -2.1643673382201136, 1.160172939300537, -1.7588465849505823);
    rw::math::Q q2(6, 0.5958089828491211, -0.5720837873271485, 1.3198588530169886, -2.1279221973814906, 0.7736911773681641, -2.633404795323507);
    rw::math::Q q3(6, 0.8070425987243652, -0.9067686361125489, 1.3945773283587855, -1.5328548711589356, 0.9157900810241699, -3.199463669453756);
    rw::math::Q q4(6, 0.7325644493103027, -0.9267538350871583, 1.7901080290423792, -2.11643185238027, 1.0915303230285645, -4.623188320790426);

    URVrepSim simHandle();
    


}