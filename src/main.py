from CoppeliaSim.CoppeliaSim import CoppeliaSim
from RobotControl.RobotControl import RobotControl
import numpy as np


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
    robot0 = RobotControl("24.5.19.20")
    robot1 = RobotControl("24.5.19.10")

    robot1.


def main():
    #coppeliaSimTest()



if __name__ == "__main__":
    main()
