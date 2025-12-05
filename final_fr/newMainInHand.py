
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


def run_async_in_thread(coro):
    def runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(coro)
        loop.close()

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    return t

async def main():
    try:
        await newEyeInHand.TrackerInitialize()
        wait_time = 0.05
        maxVel = 6.5
        step_size = 13 ## cm
        ballColor = 'y'
        start = None
        inRange = False
        x = np.zeros((5,1))
        x[4,0] = -200 ## initial gravity guess
        P = np.eye(5)

        await motor_control.init_motors()
        await motor_control.move_to_positions([0.3, 1.3, 1.3, 0.3])

        run_async_in_thread(newEyeInHand.track_loop(ballColor))
        
        # newEyeInHand.Calibrate()\
        height, width = await newEyeInHand.getCameraSize()
        print("camera height =", height)
        print("camera width =", width)
        cx = width / 2 
        cy = height / 2 

        ## can update start condidtion to do stuff
        count = 0 
        while (count < 3):
            if (start is None):
                count = 0
            start = newEyeInHand.global_u
            await asyncio.sleep(0.1)
            count += 1
        
        
        new_pos = (start, newEyeInHand.global_v)
        prev_time = time.time()
        count = 0
        start_time = prev_time
        while True:
            if new_pos[0] is not None:
                new_u, new_v = new_pos
                count = 0
            else:
                new_u = None
                new_v = None
                count += 1
            err_x = (x[0,0]) /width
            err_y = (x[1,0]) / height
            if (new_u is not None):
                if (newEyeInHand.globalRadius is not None and newEyeInHand.globalRadius < 8):
                    print(newEyeInHand.globalRadius)
                    u = 0
                    v = 0
                else:
                    u = -2*(new_u - cx) / width +0.1
                    v = -2*(new_v -cy) / height -0.05
                # print("Actual position =",u,v)
                # print("Camera position:", u,v, "Step size:", step_size, "Wait time:", wait_time)
                inRange, positions, velocities = tracker_kinematics.tracker_step(u, v, step_size, wait_time, maxVel)
            if(inRange and new_u is not None):
                await motor_control.move_to_positions(positions, velocities, wait_time *1.25)
                
            # else:
            #     print("Out of Range")
            new_pos = (newEyeInHand.global_u, newEyeInHand.global_v)
            await asyncio.sleep(0)
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
    
