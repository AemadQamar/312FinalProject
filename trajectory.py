## Handles calculating trajectory of ball
import numpy as np
import cameras
import time

class TrajectoryPlanner:
    def __init__(self):
        self.endpoint = None
        self.tracker = None

    def initializeTracker(self):
        self.tracker = cameras.Tracker('b')
        input("Press Enter to Begin Calibration")
        self.tracker.Calibrate()
    
    def endPointFinder(self, iterations):
        self.tracker.left_point = None ## reset ball positions
        self.tracker.right_point = None
        time.sleep(0.02)
        started = 0 
        while started < 2:
            points = []
            time_arr = []
            start_time = time.time()
            started = 0
            for i in range(3):
                pos = self.tracker.getAccurateBallPosition()
                if (pos is None):
                    continue
                else:
                    started += 1
                    points.append(pos)
                    time_arr.append(time.time()-start_time)
                time.sleep(0.02)
        print("throw started.")

        for i in range(3):
            if (points[i] is None):
                points.pop(i)

        while len(points) < iterations:
            pos = self.tracker.getAccurateBallPosition()
            if (pos is None):
                continue
            points.append(pos)
            time_arr.append(time.time()-start_time)
            if (len(points) > 3):
                self.endpoint = self.findEndPoint(points, time_arr)
        print("Final Endpoint found:", self.endpoint)

    def findEndPoint(self,points, time_arr): ## points and their associated times

        # guess velocities - Camera position dependent -> could use projectile motion or Kalman Filters
        xList = [p[0] for p in points]
        yList = [p[1] for p in points]
        zList = [p[2] for p in points]
        vx, x = np.polyfit(time_arr,xList,1) ## linear motion
        ay, vy, y = np.polyfit(time_arr,yList,2) ## quadratic motion
        vz, z = np.polyfit(time_arr,zList,1) ## linear motion


        n = np.array([0,1,0]) ## plane normal vector
        r = np.array([0,0,0]) ## plane reference point

        a = ay * n[1]
        b = vx*n[0] + vy*n[1] + vz*n[2]
        c = (x-r[0])*n[0] + (y-r[1])*n[1] + (z-r[2])*n[2]

        discriminant = b**2 -4*a*c
        if (discriminant < 0):
            print("Trajectory does not intersect with plane")
            return None ## no intersection with plane

        t= (-b + np.sqrt(b**2 -4*a*c)) / (2*a) ## t>0

        end_x = x + vx *t
        end_y = y + vy*t + ay* t**2
        end_z = z + vz *t
        print(a, b, c)
        return (np.array([end_x,end_y,end_z]))

if __name__ == "__main__":
    planner = TrajectoryPlanner()
    planner.initializeTracker()
    while 1:
        planner.endPointFinder(20)