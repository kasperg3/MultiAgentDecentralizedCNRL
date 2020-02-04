import numpy
import rtde_control
import rtde_receive
import rtde_io
import time, json
import logging
import rootpath

class RobotControl:

    def __init__(self, ip):
        self.rtde_c = None
        self.rtde_r = None
        self.rtde_i = None
        try:
            self.rtde_c = rtde_control.RTDEControlInterface(ip)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(ip)
            self.rtde_i = rtde_io.RTDEIOInterface(ip)
        except RuntimeError:
            logging.error("[RobotControl] Cannot connect to Universal robot")

        self.velocity = 0.5
        self.acceleration = 2

    def isConnected(self):
        if not self.rtde_c.isConnected() or not self.rtde_r.isConnected():
            return False
        return True

    def moveRobot(self, q):
        self.rtde_c.moveJ(q, self.velocity, self.acceleration)

        while not self.destinationReached(q):
            if not self.isConnected():
                self.reconnect()

            self.rtde_c.moveJ(q, self.velocity, self.acceleration)

        return True


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
