import trajectory
import asyncio
import inverse_kin
import numpy as np
import argparse
import math
import moteus
import time

transport = None
servos = None
lengths_origin = [0,0,0] ## measure this
length_per_position = 1 # m/position unit , measure this

def calculateMotorPositions(endpoint):
    rope_lengths = inverse_kin.rope_lengths(endpoint)
    positions = (rope_lengths - lengths_origin) / length_per_position
    return positions


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

async def moveMotors(positions):
    global transport, servos
    if len(servos) != 4:
        print("Could not find all motors")
        return

    await transport.cycle([x.make_stop() for x in servos])
    ## give motor commands
    commands = []
    for i in range(4):
        commands.append(servos[i].make_position(position=positions[i])) ## can limit vel
    await transport.cycle(commands)

async def main():

    iterations = 40
    planner = trajectory.TrajectoryPlanner()
    planner.initializeTracker()
    planner.endPointFinder(iterations) ## initialize finder

    await initMotors()

    for i in range(iterations):
        positions = calculateMotorPositions(planner.endpoint)
        await moveMotors(positions)
        await asyncio.sleep(0.02) ## 50Hz


if __name__ == "__main__":
    asyncio.run(main)