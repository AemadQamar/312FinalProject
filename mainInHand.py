
import asyncio
import inverse_kin
import numpy as np
import argparse
import math
import moteus
import time
import eyeInHand
import movement

transport = None
servos = None
lengths_origin = [0,0,0] ## measure this
positions_per_meter = 1 # m/position unit , measure this

def calculateMotorPositions(err_x, err_y):
    scale = 0.2
    err_x *= scale
    err_y *= scale
    # delta_rope_lengths = inverse_kin.rope_lengths([err_x, err_y]) ## rope_lengths should calculate kinematics to change by that amount
    # positions = delta_rope_lengths* positions_per_meter
    positions = None
    return positions

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
    


async def moveMotors(positions, velocities):
    global transport, servos
    ## give motor commands
    commands = []
    for i in range(4):
        commands.append(servos[i].make_position(position=positions[i], velocity_limit=0.5, watchdog_timeout=math.nan))
    c =await transport.cycle(commands)
    await asyncio.sleep(1)
    

async def main():
    ballColor = 'b'
    x = np.zeros((5,1))
    x[4,0] = -200 ## initial gravity guess
    P = np.eye(5)

    # await initMotors()
    await eyeInHand.TrackerInitialize()
    height,width = eyeInHand.getCameraSize()
    print("camera height =", height)
    print("camera width =", width)
    cx = width / 2 
    cy = height / 2 

    ## can update start condidtion to do stuff
    start = await eyeInHand.getNextBallInstance(ballColor)
    while (start is None):
        start = await eyeInHand.getNextBallInstance(ballColor)
    
    new_pos = start
    prev_time = time.time()
    count = 0
    while count < 5:
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
        err_x = cx - x[0,0]
        err_y = cy - x[1,0]
        if (new_u is not None):
            print("Actual position =", cx - new_u, cy - new_v)
        print("Predicted position =", err_x, err_y)
        print(x)

        # positions = calculateMotorPositions(err_x,err_y)
        # await moveMotors(positions)
        new_pos = await eyeInHand.getNextBallInstance(ballColor)

async def mainTest():
    await initMotors()
    positions = [0,  0,  0,  0]
    velocities = [0.5,0.5,0.5,0.5]
    await moveMotors(positions, velocities)
    # waypoints = [
    #     [0.2,   0.4,   0.4,   0.2],
    #     [2.07,  0.2,  -1.47,  0.11],
    #     [3.95, -0.28, -3.35, -0.26],
    #     [5.82, -1.0,  -5.23, -0.88],
    #     [7.69, -1.93, -7.1,  -1.74]
    # ]
    # current = waypoints[0]
    # for k in range(1, len(waypoints)):
    #     target = waypoints[k]
    #     print(f"Segment {k}: {current} → {target}")
    #     velocities = movement.compute_velocities(current, target, 2, 0.5)
    #     await moveMotors(target, velocities)
    #     await asyncio.sleep(0.2)
    #     current = target


if __name__ == "__main__":
    asyncio.run(main())