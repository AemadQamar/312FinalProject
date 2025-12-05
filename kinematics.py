# GUI
# kinematics.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import math

# ----- Frame & effector parameters -----
W = 175.0   # frame width  (cm)
H = 185.0   # frame height (cm)
d = 18.5    # end effector size (cm)
r_spool = 3.0/2   # spool radius (cm)

offset1 = .2
offset2 = 1.2
offset3 = 1.2
offset4 = .2
# Frame corners
frame_points = np.array([
    [-W/2,  H/2],  # top-left
    [ W/2,  H/2],  # top-right
    [ W/2, -H/2],  # bottom-right
    [-W/2, -H/2],  # bottom-left
])

def compute_local_eff_points(size):
    s = size / 2
    return np.array([
        [-s,  s],  # top-left
        [ s,  s],  # top-right
        [ s, -s],  # bottom-right
        [-s, -s],  # bottom-left
    ])


# ----- Global state -----
x_center = 0.0
y_center = 0.0
d_current = d
local_eff_points = compute_local_eff_points(d_current)
dragging = False
r_current = r_spool

# mode:
# 0 = absolute lengths
# 1 = delta lengths vs center (0,0)
# 2 = motor revolutions vs center (0,0) using spool radius
mode = 2

# GUI globals
fig = None
ax = None
eff_line = None
rope_lines = None
legend_text = None
x_box = None
y_box = None
d_box = None
r_box = None
mode_button = None


def compute_eff_points():
    return local_eff_points + np.array([x_center, y_center])


def rope_lengths(frame_pts, eff_pts):
    diffs = eff_pts - frame_pts
    return np.linalg.norm(diffs, axis=1)


def center_rope_lengths():
    local = compute_local_eff_points(d_current)
    eff_center = local + np.array([0.0, 0.0])
    return rope_lengths(frame_points, eff_center)


def lengths_to_turns(lengths):
    global r_current
    ref = center_rope_lengths()
    dL = lengths - ref

    circumference = 2 * math.pi * r_current
    turns = dL / circumference

    return -dL, -turns


def mode_name():
    if mode == 0:
        return "absolute lengths"
    if mode == 1:
        return "ΔL vs center"
    if mode == 2:
        return "revolutions vs center"
    return "unknown"


def format_legend(lengths):
    if mode == 0:
        # absolute lengths
        return "\n".join([
            f"Rope 1 (TR):       {lengths[1]:7.3f} cm",
            f"Rope 2 (BR):       {lengths[2]:7.3f} cm",
            f"Rope 3 (BL):       {lengths[3]:7.3f} cm",
            f"Rope 4 (TL):       {lengths[0]:7.3f} cm",
            f"Effector size:     {d_current:7.2f} cm",
            f"Spool radius:      {r_current:7.2f} cm",
            f"Mode: {mode_name()}",
        ])
    elif mode == 1:
        # delta lengths vs center
        ref = center_rope_lengths()
        dL = lengths - ref
        return "\n".join([
            f"ΔL 1 (TR):         {dL[1]:+7.3f} cm",
            f"ΔL 2 (BR):         {dL[2]:+7.3f} cm",
            f"ΔL 3 (BL):         {dL[3]:+7.3f} cm",
            f"ΔL 4 (TL):         {dL[0]:+7.3f} cm",
            f"Effector size:     {d_current:7.2f} cm",
            f"Spool radius:      {r_current:7.2f} cm",
            f"Mode: {mode_name()}",
        ])
    else:
        # mode == 2: revolutions
        dL, turns = lengths_to_turns(lengths)
        return "\n".join([
            f"Motor 1 (TR): {(turns[1]+offset1):+8.4f}  (ΔL={dL[1]:+6.2f} cm)",
            f"Motor 2 (BR): {(turns[2]+offset2):+8.4f}  (ΔL={dL[2]:+6.2f} cm)",
            f"Motor 3 (BL): {(turns[3]+offset3):+8.4f}  (ΔL={dL[3]:+6.2f} cm)",
            f"Motor 4 (TL): {(turns[0]+offset4):+8.4f}  (ΔL={dL[0]:+6.2f} cm)",
            f"Effector size:     {d_current:7.2f} cm",
            f"Spool radius:      {r_current:7.2f} cm",
            f"Mode: {mode_name()}",
        ])



def create_gui():
    global fig, ax, eff_line, rope_lines, legend_text
    global x_box, y_box, d_box, r_box, mode_button

    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.32, right=0.70)

    # Draw frame
    frame_closed = np.vstack([frame_points, frame_points[0]])
    ax.plot(frame_closed[:, 0], frame_closed[:, 1], '-k')

    eff_points = compute_eff_points()
    lengths = rope_lengths(frame_points, eff_points)

    # Draw end effector
    eff_closed = np.vstack([eff_points, eff_points[0]])
    (eff_line,) = ax.plot(eff_closed[:, 0], eff_closed[:, 1], '-b')

    # Draw ropes
    rope_lines = []
    for i in range(4):
        (line,) = ax.plot(
            [frame_points[i, 0], eff_points[i, 0]],
            [frame_points[i, 1], eff_points[i, 1]],
            '--', color='gray'
        )
        rope_lines.append(line)

    # Legend on the right
    legend_text = ax.text(
        1.05, 0.5,
        format_legend(lengths),
        transform=ax.transAxes,
        fontsize=10,
        va='center',
        ha='left',
        family='monospace'
    )

    ax.set_aspect('equal', 'box')
    margin = max(W, H) * 0.1
    ax.set_xlim(-W/2 - margin, W/2 + margin)
    ax.set_ylim(-H/2 - margin, H/2 + margin)
    ax.grid(True)

    # Text boxes: x, y, size
    axbox_x = plt.axes([0.08, 0.05, 0.18, 0.05])
    axbox_y = plt.axes([0.31, 0.05, 0.18, 0.05])
    axbox_d = plt.axes([0.54, 0.05, 0.18, 0.05])

    x_box = TextBox(axbox_x, 'x ', initial=f"{x_center:.2f}")
    y_box = TextBox(axbox_y, 'y ', initial=f"{y_center:.2f}")
    d_box = TextBox(axbox_d, 'size ', initial=f"{d_current:.2f}")

    x_box.on_submit(on_x_submit)
    y_box.on_submit(on_y_submit)
    d_box.on_submit(on_d_submit)

    # Text box: spool radius
    axbox_r = plt.axes([0.08, 0.13, 0.18, 0.05])
    r_box = TextBox(axbox_r, 'radius ', initial=f"{r_current:.2f}")
    r_box.on_submit(on_r_submit)

    # Toggle button for mode (cycles 0 -> 1 -> 2 -> 0 ...)
    ax_button = plt.axes([0.31, 0.13, 0.41, 0.05])
    mode_button = Button(ax_button, 'Mode: length / ΔL / turns')
    mode_button.on_clicked(on_mode_toggle)

    # Mouse interactions
    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    return fig


def update_plot():
    eff_points = compute_eff_points()
    lengths = rope_lengths(frame_points, eff_points)

    eff_closed = np.vstack([eff_points, eff_points[0]])
    eff_line.set_data(eff_closed[:, 0], eff_closed[:, 1])

    for i, line in enumerate(rope_lines):
        line.set_data(
            [frame_points[i, 0], eff_points[i, 0]],
            [frame_points[i, 1], eff_points[i, 1]]
        )

    legend_text.set_text(format_legend(lengths))

    fig.canvas.draw_idle()


# ----- TextBox callbacks -----
def on_x_submit(text):
    global x_center
    try:
        x_center = float(text)
        update_plot()
    except Exception:
        pass


def on_y_submit(text):
    global y_center
    try:
        y_center = float(text)
        update_plot()
    except Exception:
        pass


def on_d_submit(text):
    global d_current, local_eff_points
    try:
        d_current = float(text)
        local_eff_points = compute_local_eff_points(d_current)
        update_plot()
    except Exception:
        pass


def on_r_submit(text):
    global r_current
    try:
        r_current = float(text)
        update_plot()
    except Exception:
        pass


# ----- Mode toggle callback -----
def on_mode_toggle(event):
    global mode
    mode = (mode + 1) % 3  # 0 -> 1 -> 2 -> 0 ...
    update_plot()


# ----- Dragging callbacks -----
def on_mouse_press(event):
    global dragging
    if event.inaxes != ax:
        return
    if event.xdata is None or event.ydata is None:
        return
    dx = event.xdata - x_center
    dy = event.ydata - y_center
    if np.hypot(dx, dy) < d_current:
        dragging = True


def on_mouse_release(event):
    global dragging
    dragging = False


def on_mouse_move(event):
    global x_center, y_center
    if not dragging or event.inaxes != ax:
        return
    if event.xdata is None or event.ydata is None:
        return

    margin = d_current / 2.0
    # Clamp inside rectangular frame
    x_center = np.clip(event.xdata, -W/2 + margin, W/2 - margin)
    y_center = np.clip(event.ydata, -H/2 + margin, H/2 - margin)

    x_box.set_val(f"{x_center:.2f}")
    y_box.set_val(f"{y_center:.2f}")

    update_plot()


if __name__ == "__main__":
    fig = create_gui()
    print("Interactive ropebot GUI with:")
    print("- rectangular frame")
    print("- 3 modes: lengths, ΔL vs center, revolutions vs center")
    print("- editable effector size and spool radius")
    plt.show()
