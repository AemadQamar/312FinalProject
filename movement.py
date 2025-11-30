import asyncio
import argparse
import math
import moteus

transport = None
servos = None

NUM_MOTORS = 4

# Duration for each movement segment (time from waypoint[k-1] → waypoint[k])
SEGMENT_TIME = 2      # seconds
MAX_VEL = 0.5         # maximum allowed velocity per motor (units per second)


async def init_motors():
    """
    Initialize the moteus transport layer and discover 4 controllers.
    """
    global transport, servos
    parser = argparse.ArgumentParser()
    moteus.make_transport_args(parser)
    args = parser.parse_args([]) ## Do i need the empty list?
    transport = moteus.get_singleton_transport(args)
    devices = await transport.discover()
    addresses = [x.address for x in devices]
    servos = [moteus.Controller(id=address, transport=transport)
              for address in addresses]


def compute_velocities(prev_pos, next_pos, segment_time, max_vel=None):
    """
    Compute velocity limits for each motor so that all motors reach their
    target positions in the same amount of time (segment_time).

    Velocity_i = |next - prev| / segment_time

    Optionally clamp overall velocities if one would exceed max_vel.
    """
    if segment_time <= 0:
        raise ValueError("segment_time must be > 0")

    velocities = []
    for p, n in zip(prev_pos, next_pos):
        distance = abs(n - p)
        if distance == 0:
            velocities.append(0.0)
        else:
            velocities.append(distance / segment_time)

    # If the largest velocity exceeds max_vel, scale all of them down
    if max_vel is not None:
        vmax = max(velocities)
        if vmax > max_vel:
            scale = max_vel / vmax
            velocities = [v * scale for v in velocities]

    return velocities


async def move_segment(prev_pos, next_pos, segment_time=SEGMENT_TIME, max_vel=MAX_VEL):
    """
    Send a single synchronized movement command for all motors:
    move from prev_pos to next_pos in segment_time seconds.

    Each motor receives its own velocity_limit so that all motors
    finish at approximately the same time.
    """
    global transport, servos

    velocities = compute_velocities(prev_pos, next_pos, segment_time, max_vel)

    commands = []
    for i in range(NUM_MOTORS):
        cmd = servos[i].make_position(
            position=float(next_pos[i]),
            velocity=0.0,                       # we do not specify target velocity
            velocity_limit=float(velocities[i]),
            accel_limit=1.0
        )
        commands.append(cmd)

    # Execute all commands simultaneously
    await transport.cycle(commands)

    # Wait for the movement to complete before continuing
    await asyncio.sleep(segment_time)


async def follow_trajectory(waypoints, segment_time=SEGMENT_TIME, max_vel=MAX_VEL):
    """
    Execute a waypoint trajectory.

    waypoints is a list of position vectors [m1, m2, m3, m4].
    The first waypoint is the robot's current real position,
    so we start moving from waypoint[1] onward.
    """

    if len(waypoints) < 2:
        print("Not enough waypoints to move")
        return

    # The first waypoint is the assumed starting position
    current = waypoints[0]

    # Move through all remaining waypoints
    for k in range(1, len(waypoints)):
        target = waypoints[k]
        print(f"Segment {k}: {current} → {target}")
        await move_segment(current, target, segment_time, max_vel)
        current = target


async def main():
    # Example waypoint list
    waypoints = [
        [0.2,   0.4,   0.4,   0.2],
        [2.07,  0.2,  -1.47,  0.11],
        [3.95, -0.28, -3.35, -0.26],
        [5.82, -1.0,  -5.23, -0.88],
        [7.69, -1.93, -7.1,  -1.74]
    ]

    await follow_trajectory(waypoints, segment_time=0.2, max_vel=10.0)


if __name__ == "__main__":
    asyncio.run(main())