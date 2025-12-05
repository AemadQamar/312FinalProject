
import asyncio
import threading
import tracker_kinematics
import numpy as np
import argparse
import math
import moteus
import time
import motor_control
import newEyeInHand

transport = None
servos = None
lengths_origin = [0,0,0] ## measure this
positions_per_meter = 1 # m/position unit , measure this


async def main():
    try:
        wait_time = 0.05
        maxVel = 3
        step_size = 13 ## cm
        ballColor = 'y'
        start = None
        inRange = False
        x = np.zeros((5,1))
        x[4,0] = -200 ## initial gravity guess
        P = np.eye(5)

        await motor_control.init_motors()
        await motor_control.move_to_positions([0.3, 1.3, 1.3, 0.3])
        
        while True:
            u,v = float(input("toss x: ").split("0"))
            v = float(input("toss y: "))
            print("1")
            inRange, positions, velocities = tracker_kinematics.tracker_step(u, v, step_size, wait_time, maxVel, True)
            print(positions, velocities)
            await motor_control.move_to_positions(positions, velocities, wait_time *1.25)
            print("3")
    except Exception as e:
        print("Error:", e)
    finally:
        print("Stopping Motors")
        await motor_control.move_to_positions([0.3, 1.3, 1.3, 0.3])
        i = input("Cut Motors?")
        if ( i != "n"):
            await motor_control.stop_motors()
if __name__ == "__main__":
    asyncio.run(main())
    
