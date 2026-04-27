import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons

from knob import Knob
from ur5_robot import UR5Robot


class UR5Visualizer:
    def __init__(self):
        self.robot = UR5Robot()
        self.fig = plt.figure(figsize=(26, 14))

        self.ax = self.fig.add_axes([0.02, 0.26, 0.72, 0.76], projection='3d')

        self.ax_info = self.fig.add_axes([0.65, 0.20, 0.97, 0.78])
        self.ax_info.axis('off')

        ax_mode = plt.axes([0.65, 0.10, 0.18, 0.10])
        self.radio_mode = RadioButtons(ax_mode, ('Forward K', 'Trajectory'))
        self.control_mode = 'Forward K'
        self.radio_mode.on_clicked(self.set_control_mode)
        for label in self.radio_mode.labels:
            label.set_fontsize(20)

        knob_size = 0.10
        knob_spacing_x = 0.09
        knob_spacing_y = 0.12
        knob_start_x = 0.08
        knob_start_y = 0.03

        self.fk_knobs = []
        for i in range(6):
            col = i % 3
            row = 1 - (i // 3)
            ax_knob = plt.axes([knob_start_x + col * knob_spacing_x,
                                 knob_start_y + row * knob_spacing_y,
                                 knob_size, knob_size])
            knob = Knob(ax_knob, f'θ{i + 1}', -180, 180, 0)
            knob.on_changed(self.update_fk)
            self.fk_knobs.append(knob)
            ax_knob.set_visible(True)

        slider_left_t = 0.10
        slider_width_t = 0.30
        slider_height_t = 0.025
        slider_spacing_t = 0.035
        slider_bottom_t = 0.14

        home_pos = self.robot.current_pos

        ax_tx = plt.axes([slider_left_t, slider_bottom_t + 0 * slider_spacing_t, slider_width_t, slider_height_t])
        ax_ty = plt.axes([slider_left_t, slider_bottom_t + 1 * slider_spacing_t, slider_width_t, slider_height_t])
        ax_tz = plt.axes([slider_left_t, slider_bottom_t + 2 * slider_spacing_t, slider_width_t, slider_height_t])

        slider_tx = Slider(ax_tx, 'Tgt X', -1.0, 1.0, valinit=home_pos[0], valstep=0.01, color='lightcoral')
        slider_ty = Slider(ax_ty, 'Tgt Y', -1.0, 1.0, valinit=home_pos[1], valstep=0.01, color='lightcoral')
        slider_tz = Slider(ax_tz, 'Tgt Z', -0.5, 1.0, valinit=home_pos[2], valstep=0.01, color='lightcoral')

        for s in [slider_tx, slider_ty, slider_tz]:
            s.label.set_fontsize(16)
            s.valtext.set_fontsize(14)

        self.traj_sliders = [slider_tx, slider_ty, slider_tz]
        for slider_ax in [ax_tx, ax_ty, ax_tz]:
            slider_ax.set_visible(False)

        ax_traj_type = plt.axes([slider_left_t, 0.03, 0.16, 0.08])
        self.radio_traj = RadioButtons(ax_traj_type, ('Line', 'Square', 'Circle'))
        self.traj_type = 'Line'
        self.radio_traj.on_clicked(self.set_traj_type)
        for label in self.radio_traj.labels:
            label.set_fontsize(14)
        ax_traj_type.set_visible(False)
        self.ax_traj_type = ax_traj_type

        ax_plane = plt.axes([slider_left_t + 0.18, 0.03, 0.15, 0.08])
        self.radio_plane = RadioButtons(ax_plane, ('XY', 'XZ', 'YZ'))
        self.plane = 'xy'
        self.radio_plane.on_clicked(self.set_plane)
        for label in self.radio_plane.labels:
            label.set_fontsize(14)
        ax_plane.set_visible(False)
        self.ax_plane = ax_plane

        ax_execute = plt.axes([0.44, 0.18, 0.12, 0.06])
        self.btn_execute = Button(ax_execute, 'Execute')
        self.btn_execute.on_clicked(self.execute_trajectory)
        self.btn_execute.label.set_fontsize(16)
        ax_execute.set_visible(False)
        self.ax_execute = ax_execute

        ax_reset = plt.axes([0.44, 0.11, 0.12, 0.06])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset_robot)
        self.btn_reset.label.set_fontsize(16)

        self.trajectory = []
        self.trajectory_positions = None
        self.singularity_flags = []
        self.traj_index = 0
        self.animating = False
        self.timer = None
        self.all_trajectories = []
        self.trajectory_blocked = False

        self.status_text = "Ready"
        self.is_singular = False
        self.manipulability = 0.0
        self.condition_number = 0.0

        self.update_plot()
        plt.show()

    def set_control_mode(self, label):
        self.control_mode = label

        for knob in self.fk_knobs:
            knob.ax.set_visible(False)
        for slider in self.traj_sliders:
            slider.ax.set_visible(False)
        self.ax_traj_type.set_visible(False)
        self.ax_plane.set_visible(False)
        self.ax_execute.set_visible(False)

        if label == 'Forward K':
            for knob in self.fk_knobs:
                knob.ax.set_visible(True)
        else:
            for slider in self.traj_sliders:
                slider.ax.set_visible(True)
            self.ax_traj_type.set_visible(True)
            self.ax_plane.set_visible(True)
            self.ax_execute.set_visible(True)

        plt.draw()

    def set_traj_type(self, label):
        self.traj_type = label

    def set_plane(self, label):
        self.plane = label.lower()

    def update_fk(self, val):
        for i in range(6):
            self.robot.theta[i] = np.radians(self.fk_knobs[i].val)

        self.robot.current_pos = self.robot.forward_kinematics(self.robot.theta)[:3, 3]

        is_singular, manip, cond = self.robot.check_singularity(self.robot.theta)
        self.is_singular = is_singular
        self.manipulability = manip
        self.condition_number = cond
        self.status_text = "WARNING: Singularity Detected!" if is_singular else "Forward Kinematics Mode"

        self.update_plot()

    def execute_trajectory(self, event):
        if self.animating:
            return

        target_pos = np.array([s.val for s in self.traj_sliders])

        if self.traj_type == 'Line':
            self.trajectory, self.trajectory_positions, self.singularity_flags = self.robot.move_line(target_pos)
        elif self.traj_type == 'Square':
            traj_line, _, _ = self.robot.move_line(target_pos)
            if traj_line:
                self.robot.theta = traj_line[-1]
                self.robot.current_pos = self.robot.forward_kinematics(self.robot.theta)[:3, 3]
            self.trajectory, self.trajectory_positions, self.singularity_flags = self.robot.move_square(size=0.1, plane=self.plane)
        else:
            traj_line, _, _ = self.robot.move_line(target_pos)
            if traj_line:
                self.robot.theta = traj_line[-1]
                self.robot.current_pos = self.robot.forward_kinematics(self.robot.theta)[:3, 3]
            self.trajectory, self.trajectory_positions, self.singularity_flags = self.robot.move_circle(radius=0.1, plane=self.plane)

        if self.trajectory:
            self.traj_index = 0
            self.animating = True
            self.trajectory_blocked = False
            self.animate()

    def draw_coordinate_frame(self, T, scale=0.08):
        origin = T[:3, 3]
        for axis_idx, color in enumerate(['r-', 'b-', 'g-']):
            end = origin + T[:3, axis_idx] * scale
            self.ax.plot([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]], color, linewidth=4)

    def draw_end_effector(self, T_ee):
        ee_pos = T_ee[:3, 3]
        self.ax.scatter([ee_pos[0]], [ee_pos[1]], [ee_pos[2]],
                        c='orange', s=500, marker='*', edgecolors='black', linewidths=4, zorder=10)

    def update_info_display(self):
        self.ax_info.clear()
        self.ax_info.axis('off')

        bg_color = 'lightsalmon' if self.is_singular else 'lightblue'
        T_ee = self.robot.forward_kinematics(self.robot.theta)

        info_text = "WARNING: SINGULARITY DETECTED\n" if self.is_singular else ""
        info_text += f"Mode: {self.control_mode}\n"
        info_text += "=" * 40 + "\n\n"
        info_text += "Joint Angles:\n"
        for i in range(6):
            info_text += f"  θ{i + 1}: {np.degrees(self.robot.theta[i]):>8.2f}°\n"
        info_text += "\n" + "=" * 40 + "\n"
        info_text += "End Effector Transform:\n"
        for i in range(4):
            info_text += "[" + "".join(f"{T_ee[i, j]:>7.3f}" + (" " if j < 3 else "") for j in range(4)) + "]\n"
        info_text += "\n" + "=" * 40 + "\n"
        info_text += "Position (m):\n"
        info_text += f"  x = {T_ee[0, 3]:>9.4f}\n"
        info_text += f"  y = {T_ee[1, 3]:>9.4f}\n"
        info_text += f"  z = {T_ee[2, 3]:>9.4f}\n"
        info_text += "\n" + "=" * 40 + "\n"
        info_text += "Singularity Analysis:\n"
        info_text += f"  Manip: {self.manipulability:.6f}\n"
        info_text += f"  Cond#: {self.condition_number:.2f}\n"
        if self.trajectory_blocked:
            info_text += "\nTRAJ BLOCKED AT SINGULARITY\n"
        info_text += f"\nStatus: {self.status_text}\n"

        self.ax_info.text(0.0, 0.5, info_text, fontsize=15, verticalalignment='center',
                          family='monospace',
                          bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.9, pad=2.5))

    def update_plot(self):
        self.ax.clear()
        positions, transforms = self.robot.get_joint_positions(self.robot.theta)

        link_color = 'red' if self.is_singular else 'navy'
        self.ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                     'o-', linewidth=6, markersize=15, color=link_color)

        for T in transforms:
            self.draw_coordinate_frame(T, scale=0.04)
        self.draw_end_effector(transforms[-1])

        for traj_pos in self.all_trajectories:
            if len(traj_pos) > 0:
                self.ax.plot(traj_pos[:, 0], traj_pos[:, 1], traj_pos[:, 2],
                             ':', linewidth=4, color='black', alpha=0.7)

        if self.trajectory_positions is not None and len(self.trajectory_positions) > 0:
            if self.traj_index < len(self.trajectory_positions):
                remaining = self.trajectory_positions[self.traj_index:]
                self.ax.plot(remaining[:, 0], remaining[:, 1], remaining[:, 2],
                             '--', linewidth=4, color='black', alpha=0.8)
            if self.traj_index > 0:
                completed = self.trajectory_positions[:self.traj_index + 1]
                self.ax.plot(completed[:, 0], completed[:, 1], completed[:, 2],
                             ':', linewidth=4, color='black', alpha=0.9)

        self.ax.set_xlabel('X (m)', fontsize=16, weight='bold')
        self.ax.set_ylabel('Y (m)', fontsize=16, weight='bold')
        self.ax.set_zlabel('Z (m)', fontsize=16, weight='bold')
        self.ax.set_xlim([-0.6, 0.6])
        self.ax.set_ylim([-0.6, 0.6])
        self.ax.set_zlim([-0.3, 0.7])
        self.ax.tick_params(labelsize=13)

        self.update_info_display()
        self.fig.canvas.draw()

    def reset_robot(self, event):
        if self.animating:
            return
        self.animating = False
        if self.timer is not None:
            self.timer.stop()
            self.timer = None

        self.robot.reset_to_home()
        for knob in self.fk_knobs:
            knob.set_val(0)

        home_pos = self.robot.current_pos
        for i, slider in enumerate(self.traj_sliders):
            slider.set_val(home_pos[i])

        self.all_trajectories = []
        self.trajectory_positions = None
        self.trajectory = []
        self.singularity_flags = []
        self.traj_index = 0
        self.trajectory_blocked = False
        self.status_text = "Reset to home"
        self.is_singular = False
        self.update_plot()

    def animate(self):
        if self.animating and self.traj_index < len(self.trajectory):
            if self.traj_index < len(self.singularity_flags) and self.singularity_flags[self.traj_index]:
                self.animating = False
                self.trajectory_blocked = True
                self.is_singular = True
                _, self.manipulability, self.condition_number = self.robot.check_singularity(self.robot.theta)
                self.status_text = "WARNING: Trajectory stopped at singularity!"
                self.update_plot()
                return

            self.robot.theta = self.trajectory[self.traj_index]
            self.robot.current_pos = self.robot.forward_kinematics(self.robot.theta)[:3, 3]

            self.is_singular, self.manipulability, self.condition_number = self.robot.check_singularity(self.robot.theta)
            self.status_text = f"Trajectory step {self.traj_index + 1} of {len(self.trajectory)}"
            self.update_plot()
            self.traj_index += 1

            if self.traj_index >= len(self.trajectory):
                if self.trajectory_positions is not None:
                    self.all_trajectories.append(self.trajectory_positions.copy())
                self.animating = False
                self.status_text = "Trajectory Complete"
                self.update_plot()
            else:
                self.timer = self.fig.canvas.new_timer(interval=50)
                self.timer.single_shot = True
                self.timer.add_callback(self.animate)
                self.timer.start()
