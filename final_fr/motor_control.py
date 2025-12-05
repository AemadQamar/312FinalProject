# motor_control.py

import asyncio
import argparse
import math
import moteus

NUM_MOTORS = 4

VEL_LIMIT = 1.5
ACC_LIMIT = 50
TORQUE_LIMIT = 3.0
DEFAULT_WAIT = 5.0

transport = None
servos = None


async def init_motors():
    global transport, servos

    parser = argparse.ArgumentParser()
    moteus.make_transport_args(parser)
    args = parser.parse_args([])

    transport = moteus.get_singleton_transport(args)
    devices = await transport.discover()

    print(f"Found {len(devices)} moteus devices")

    if len(devices) < NUM_MOTORS:
        raise RuntimeError(f"Expected {NUM_MOTORS} motors, found {len(devices)}")

    servos = [moteus.Controller(id=d.address, transport=transport) for d in devices]

    await transport.cycle([s.make_stop() for s in servos])

    print("Motors initialized:", [d.address for d in devices])


async def move_to_positions(target, velocities=None, wait_time=DEFAULT_WAIT):
    global servos, transport

    if servos is None or transport is None:
        raise RuntimeError("Call init_motors() first")

    if len(target) != NUM_MOTORS:
        raise ValueError("target must have 4 elements")

    if velocities is None:
        velocities = [VEL_LIMIT] * NUM_MOTORS
    elif len(velocities) != NUM_MOTORS:
        raise ValueError("velocities must have 4 elements")

    # print(f"\nMoving to {target}\n")
    # print(f"Velocities: {velocities}")

    commands = []
    for i in range(NUM_MOTORS):
        cmd = servos[i].make_position(
            position=float(target[i]),
            velocity_limit=float(velocities[i]),
            accel_limit=ACC_LIMIT,
            maximum_torque=TORQUE_LIMIT,
            watchdog_timeout=math.nan,
        )
        commands.append(cmd)

    await transport.cycle(commands)
    await asyncio.sleep(wait_time)

async def stop_motors():
    commands = []
    for i in range(4):
        commands.append(servos[i].make_stop())
    await transport.cycle(commands)