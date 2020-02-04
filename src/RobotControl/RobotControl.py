import numpy
import rtde_control
import rtde_receive
import rtde_io
import logging
from RobotControl.robotiq_gripper import RobotiqGripper

class RobotControl:

    def __init__(self, ip):
        self.rtde_c = rtde_control.RTDEControlInterface(ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ip)

        self.velocity = 0.5
        self.acceleration = 2

        #Gripper Setup
        self.gripper = RobotiqGripper(self.rtde_c)
        self.gripper.activate()     # returns to previous position after activation
        self.gripper.set_force(50)  # from 0 to 100 %
        self.gripper.set_speed(100) # from 0 to 100 %
        self.gripper.open()         # Open the gripper

    def isConnected(self):
        if not self.rtde_c.isConnected() or not self.rtde_r.isConnected():
            return False
        return True

    def moveGripper(self, pos):
        return self.gripper.move(pos)

    def moveHome(self):
        return self.moveRobot([0, -1.57, 0, -1.57, 0, 0])

    def moveRobot(self, q):
        return self.rtde_c.moveJ(q, self.velocity, self.acceleration)

    def destinationReached(self, q):
        difference = numpy.subtract(q,  self.getQ())
        for i in difference:
            if abs(i) > 0.03:
                return False
        return True

    def getRuntimeState(self):
        return self.rtde_r.getRuntimeState()

    def getSafetyMode(self):
        return self.rtde_r.getSafetyMode()

    def reconnect(self):
        if not self.rtde_r.isConnected():
            self.rtde_r.reconnect()
        if not self.rtde_c.isConnected():
            self.rtde_c.reconnect()

    def isEmergencyStopped(self):
        if self.rtde_r.getSafetyMode() == 7:
            return True
        return False

    def getQ(self):
        return self.rtde_r.getActualQ()

    def stopScript(self):
        # Stop the rtde control script
        self.rtde_c.stopRobot()
