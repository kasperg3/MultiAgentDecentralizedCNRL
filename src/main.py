import time

from CoppeliaSim.CoppeliaSim import CoppeliaSim
from RobotControl.RobotControl import RobotControl
import numpy as np
import CoppeliaSim.sim as sim


def coppeliaSimTest():
    robot0 = CoppeliaSim("0")
    robot1 = CoppeliaSim("1")
    robot0.startSim()

    robot0.setQ(np.array([-0.667, -1.84571, -2.10352, -0.758907, 1.60592, 0.903087], dtype=np.float32))
    robot1.setQ(np.array([-0.667, -1.84571, -2.10352, -0.758907, 1.60592, 0.903087], dtype=np.float32))
    robot1.setQ(np.array([-0.2, -1.84571, -2.10352, -0.758907, 1.60592, 0.903087], dtype=np.float32))

    robot0.closeGripper()
    robot1.closeGripper()

    robot1.stopSim()


def RobotControlTest():
    robot1 = RobotControl("24.5.19.20")
    robot1.moveHome()
    # robot1.stopScript()


def main():
    # coppeliaSimTest()
    # RobotControlTest()

    #Init the remote api
    remoteClientID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to CoppeliaSim
    if remoteClientID != -1:
        print("Successfully connected to Coppelia RemoteAPI")
    else:
        print('Failed connecting to remote API server')

    robot0 = CoppeliaSim("0", remoteClientID)

    print("Quat: " + robot0.getQuat('ROBOTIQ'))


if __name__ == "__main__":
    main()
