
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


def updateKalman(x, P, new_u, new_v, dt):
    """
    x: 4x1 state [u, v, u_dot, v_dot]^T
    P: 4x4 covariance
    new_u, new_v: measured ball position
    dt: time step (s)
    """
    # ---------- State Transition ----------
    F = np.array([
        [1.0, 0.0, dt,   0.0,       0.0],            # u
        [0.0, 1.0, 0.0,  dt,  0.5 * dt**2],          # v depends on g_px
        [0.0, 0.0, 1.0,  0.0,       0.0],            # u_dot
        [0.0, 0.0, 0.0,  1.0,       dt],             # v_dot depends on g_px
        [0.0, 0.0, 0.0,  0.0,       1.0]             # g_px (random walk)
    ], dtype=float)

    # Process noise
    q_pos = 1.0
    q_vel = 1.0
    q_g   = 2.0   
    Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_g])

    # Measurement model: we only measure (u, v)
    H = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0]
    ], dtype=float)

    # measurement noise (tune to detector)
    R = np.eye(2) * 0.5

    # ---------- Prediction ----------
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q

    if (new_u is None):
        return x_pred, P_pred

    # ---------- Update ----------
    z = np.array([[new_u], [new_v]], dtype=float)
    y = z - (H @ x_pred)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    x_new = x_pred + K @ y
    P_new = (np.eye(5) - K @ H) @ P_pred

    return x_new, P_new

async def initMotors():
    global transport, servos
    parser = argparse.ArgumentParser()
    moteus.make_transport_args(parser)
    args = parser.parse_args([]) ## Do i need the empty list?
    transport = moteus.get_singleton_transport(args)
    devices = await transport.discover()
    addresses = [x.address for x in devices]
    servos = [moteus.Controller(id=address, transport=transport)
              for address in addresses]
    if len(servos) != 4:
        print("Could not find all motors")
        return

    await transport.cycle([x.make_stop() for x in servos])

async def moveMotors(positions):
    global transport, servos
    ## give motor commands
    commands = []
    for i in range(4):
        commands.append(servos[i].make_position(position=positions[i])) ## can limit vel
    await transport.cycle(commands)

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
        wait_time = 0.0
        maxVel = 5.0
        step_size = 8 ## cm
        ballColor = 'y'
        start = None
        x = np.zeros((5,1))
        x[4,0] = -200 ## initial gravity guess
        P = np.eye(5)

        # await initMotors()
        await motor_control.init_motors()
        await motor_control.move_to_positions([0.2, 1.2, 1.2, 0.2])

        run_async_in_thread(newEyeInHand.track_loop('y'))
        
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
            start = newEyeInHand.latest_ball
            await asyncio.sleep(0.1)
            count += 1
        
        
        new_pos = start
        prev_time = time.time()
        count = 0
        start_time = prev_time
        while True:
            if new_pos is not None:
                new_u, new_v = new_pos
                count = 0
            else:
                new_u = None
                new_v = None
                count += 1
            dt = time.time() - prev_time
            prev_time = time.time()
            x, P = updateKalman(x, P, new_u, new_v, dt)
            err_x = (x[0,0]) /width
            err_y = (x[1,0]) / height
            if (new_u is not None):
                u = -2*(new_u - cx) / width
                v = -2*(new_v -cy) / height
                # print("Actual position =",u,v)
                print("Camera position:", u,v, "Step size:", step_size, "Wait time:", wait_time)
                inRange, positions, velocities = tracker_kinematics.tracker_step(u, v, step_size, wait_time, maxVel)
            else:
                inRange = False
                print("Predicted position =", err_x, err_y)
                inRange, positions, velocities = tracker_kinematics.tracker_step(-err_x, -err_y, step_size, wait_time, maxVel)
            if(inRange):
                # print()
                await motor_control.move_to_positions(positions, velocities, wait_time *1.25)
                # await moveMotors(positions)
            else:
                print("Out of Range")
            new_pos = newEyeInHand.latest_ball
            await asyncio.sleep(0)
    except Exception as e:
        print("Error:", e)
    finally:
        print("Stopping Motors")
        await motor_control.move_to_positions([0.3, 1.3, 1.3, 0.3])
        input("Cut Motors?")
        await motor_control.stop_motors()
if __name__ == "__main__":
    asyncio.run(main())
    
