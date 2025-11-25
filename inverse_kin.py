#Matplotlib interactive visualisation of the kinematics (doesn't seem to work in colab environment)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

# ----- Parameters -----
S = 200.0    # frame size (cm)
d = 30.0     # end effector size (cm) - INITIAL

# Rope order:
# 0: top-left
# 1: top-right
# 2: bottom-right
# 3: bottom-left

frame_points = np.array([
    [-S/2,  S/2],  # top-left
    [ S/2,  S/2],  # top-right
    [ S/2, -S/2],  # bottom-right
    [-S/2, -S/2],  # bottom-left
])


def compute_local_eff_points(size):
    """Return the 4 local corners for a given end effector size."""
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


def compute_eff_points():
    """World coordinates of the end effector corners."""
    global local_eff_points, x_center, y_center
    return local_eff_points + np.array([x_center, y_center])


def rope_lengths(frame_pts, eff_pts):
    diffs = eff_pts - frame_pts
    return np.linalg.norm(diffs, axis=1)


def format_legend(lengths):
    return "\n".join([
        f"Rope top-left:     {lengths[0]:7.3f}",
        f"Rope top-right:    {lengths[1]:7.3f}",
        f"Rope bottom-right: {lengths[2]:7.3f}",
        f"Rope bottom-left:  {lengths[3]:7.3f}",
        f"Effector size:     {d_current:7.2f} cm",
    ])


def create_gui():
    global fig, ax, eff_line, rope_lines, legend_text, x_box, y_box, d_box

    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.25, right=0.7)

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
    margin = S * 0.1
    ax.set_xlim(-S/2 - margin, S/2 + margin)
    ax.set_ylim(-S/2 - margin, S/2 + margin)
    ax.grid(True)

    # Text boxes
    axbox_x = plt.axes([0.10, 0.05, 0.20, 0.05])
    axbox_y = plt.axes([0.40, 0.05, 0.20, 0.05])
    axbox_d = plt.axes([0.70, 0.05, 0.20, 0.05])

    x_box = TextBox(axbox_x, 'x ', initial=str(x_center))
    y_box = TextBox(axbox_y, 'y ', initial=str(y_center))
    d_box = TextBox(axbox_d, 'size ', initial=str(d_current))

    x_box.on_submit(on_x_submit)
    y_box.on_submit(on_y_submit)
    d_box.on_submit(on_d_submit)

    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    return fig


def update_plot():
    """Recompute everything and redraw."""
    global eff_line, rope_lines, legend_text

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
    except:
        pass


def on_y_submit(text):
    global y_center
    try:
        y_center = float(text)
        update_plot()
    except:
        pass


def on_d_submit(text):
    """Resize the end effector live."""
    global d_current, local_eff_points
    try:
        d_current = float(text)
        local_eff_points = compute_local_eff_points(d_current)
        update_plot()
    except:
        pass


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

    margin = d_current / 2
    x_center = np.clip(event.xdata, -S/2 + margin, S/2 - margin)
    y_center = np.clip(event.ydata, -S/2 + margin, S/2 - margin)

    x_box.set_val(f"{x_center:.2f}")
    y_box.set_val(f"{y_center:.2f}")

    update_plot()


if __name__ == "__main__":
    fig = create_gui()
    print("Interactive ropebot GUI with adjustable end effector size.")
    plt.show()
