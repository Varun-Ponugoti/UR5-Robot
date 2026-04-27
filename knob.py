import numpy as np
from matplotlib.patches import Circle


class Knob:
    def __init__(self, ax, label, valmin=-180, valmax=180, valinit=0):
        self.ax = ax
        self.label = label
        self.valmin = valmin
        self.valmax = valmax
        self.val = valinit
        self.observers = []

        self.ax.clear()
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        outer_ring = Circle((0, 0), 1.1, color='darkgray', ec='black', linewidth=2.5, zorder=1)
        self.ax.add_patch(outer_ring)

        self.circle = Circle((0, 0), 1, color='silver', ec='black', linewidth=3, zorder=2)
        self.ax.add_patch(self.circle)

        inner_circle = Circle((0, 0), 0.85, color='gainsboro', ec='gray', linewidth=1.5, zorder=3)
        self.ax.add_patch(inner_circle)

        for angle in range(0, 360, 30):
            angle_rad = np.radians(angle)
            x1, y1 = 0.9 * np.cos(angle_rad), 0.9 * np.sin(angle_rad)
            x2, y2 = 1.0 * np.cos(angle_rad), 1.0 * np.sin(angle_rad)
            self.ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, zorder=4)

        angle_rad = np.radians(valinit)
        pointer_length = 0.75
        self.indicator, = self.ax.plot([0, pointer_length * np.cos(angle_rad)],
                                       [0, pointer_length * np.sin(angle_rad)],
                                       'r-', linewidth=4, zorder=5)

        self.pointer_dot, = self.ax.plot([pointer_length * np.cos(angle_rad)],
                                         [pointer_length * np.sin(angle_rad)],
                                         'ro', markersize=12, zorder=6)

        self.ax.plot(0, 0, 'ko', markersize=14, zorder=7)
        self.ax.plot(0, 0, 'wo', markersize=7, zorder=8)

        self.ax.text(0, -1.6, label, ha='center', fontsize=16, weight='bold')
        self.value_text = self.ax.text(0, 1.35, f'{valinit:.0f}°', ha='center', fontsize=14,
                                       bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8))

        self.fig = ax.figure
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        self.dragging = False

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        dx = event.xdata
        dy = event.ydata
        if dx is None or dy is None:
            return
        dist = np.sqrt(dx ** 2 + dy ** 2)
        if dist <= 1.1:
            self.dragging = True

    def on_motion(self, event):
        if not self.dragging or event.inaxes != self.ax:
            return
        dx = event.xdata
        dy = event.ydata
        if dx is None or dy is None:
            return
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        self.set_val(angle_deg)

    def on_release(self, event):
        self.dragging = False

    def set_val(self, val):
        self.val = np.clip(val, self.valmin, self.valmax)

        angle_rad = np.radians(self.val)
        pointer_length = 0.75
        self.indicator.set_data([0, pointer_length * np.cos(angle_rad)],
                                 [0, pointer_length * np.sin(angle_rad)])
        self.pointer_dot.set_data([pointer_length * np.cos(angle_rad)],
                                   [pointer_length * np.sin(angle_rad)])
        self.value_text.set_text(f'{self.val:.0f}°')

        for func in self.observers:
            func(self.val)

        self.fig.canvas.draw_idle()

    def on_changed(self, func):
        self.observers.append(func)
