# tracker_kinematics.py
# Use kinematics.py to map tracker (cx, cy) to motor targets + velocities.

import math
import numpy as np
import kinematics

center_x = 0.0
center_y = 0.0
motor_pos = [0.0, 0.0, 0.0, 0.0]  # [TR, BR, BL, TL]

offset1 = .4
offset2 = 1.4
offset3 = 1.4
offset4 = .4



def compute_velocities(prev_pos, next_pos, segment_time, max_vel=None):
    if segment_time <= 0:
        raise ValueError("segment_time must be > 0")

    v = []
    for a, b in zip(prev_pos, next_pos):
        d = abs(b - a)
        v.append(d / segment_time if d > 0 else 0.0)

    if max_vel is not None:
        vmax = max(v)
        if vmax > max_vel and vmax > 0:
            s = max_vel / vmax
            v = [vi * s for vi in v]

    return v


def tracker_step(cx, cy, step_cm=5.0, segment_time=2.0, max_vel=1.0):
    """
    cx, cy in [-1,1] -> (ok, motor_pos, velocities)

    motor_pos:  [TR, BR, BL, TL] in revolutions
    velocities: [TR, BR, BL, TL] in rev/s
    """

    global center_x, center_y, motor_pos

    dx = float(cx) * step_cm
    dy = float(cy) * step_cm

    new_x = center_x + dx
    new_y = center_y + dy

    W = kinematics.W
    H = kinematics.H
    d = kinematics.d_current
    margin = d / 2.0

    new_x = np.clip(new_x, -W/2 + margin, W/2 - margin)
    new_y = np.clip(new_y, -H/2 + margin, H/2 - margin)

    max_x = 60.0
    max_y = 60.0
    if new_x < -max_x or new_x > max_x or new_y < -max_y or new_y > max_y:
        return False, None, None

    eff_pts = kinematics.compute_local_eff_points(d) + np.array([new_x, new_y])
    lengths = kinematics.rope_lengths(kinematics.frame_points, eff_pts)

    dL, turns = kinematics.lengths_to_turns(lengths)

    new_motor_pos = [turns[1]+offset1, turns[2]+offset2, turns[3]+offset3, turns[0]+offset4]  # TR, BR, BL, TL

    vel = compute_velocities(motor_pos, new_motor_pos, segment_time, max_vel)

    center_x = new_x
    center_y = new_y
    motor_pos = new_motor_pos

    return True, new_motor_pos, vel


if __name__ == "__main__":
    print("start center:", center_x, center_y)
    print("start motors:", motor_pos)

    ok1, pos1, vel1 = tracker_step(1.0, 0.0, step_cm=5.0, segment_time=2.0, max_vel=1.0)
    print("step 1 ok:", ok1)
    print("center:", center_x, center_y)
    print("motors:", pos1)
    print("vel:", vel1)

    ok2, pos2, vel2 = tracker_step(-0.5, 0.5, step_cm=5.0, segment_time=2.0, max_vel=1.0)
    print("step 2 ok:", ok2)
    print("center:", center_x, center_y)
    print("motors:", pos2)
    print("vel:", vel2)
