import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons
from matplotlib.patches import Circle, Wedge, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D


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


class UR5Robot:
    def __init__(self):
        self.d1 = 0.0892
        self.a2 = 0.425
        self.a3 = 0.392
        self.d4 = 0.1093
        self.d5 = 0.09475
        self.d6 = 0.0825

        self.a = [0, -self.a2, -self.a3, 0, 0, 0]
        self.alpha = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]
        self.d = [self.d1, 0, 0, self.d4, self.d5, self.d6]

        self.home_theta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.theta = self.home_theta.copy()
        self.current_pos = self.forward_kinematics(self.theta)[:3, 3]

    def dh_transform(self, a, alpha, d, theta):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, theta):
        T = np.eye(4)
        for i in range(6):
            T = T @ self.dh_transform(self.a[i], self.alpha[i], self.d[i], theta[i])
        return T

    def get_joint_positions(self, theta):
        positions = [np.array([0, 0, 0])]
        transforms = [np.eye(4)]
        T = np.eye(4)

        for i in range(6):
            T = T @ self.dh_transform(self.a[i], self.alpha[i], self.d[i], theta[i])
            positions.append(T[:3, 3])
            transforms.append(T.copy())

        return np.array(positions), transforms

    def jacobian(self, theta):
        J = np.zeros((6, 6))
        T = [np.eye(4)]

        for i in range(6):
            T.append(T[-1] @ self.dh_transform(self.a[i], self.alpha[i], self.d[i], theta[i]))

        T_ee = T[-1]
        o_ee = T_ee[:3, 3]

        for i in range(6):
            z_i = T[i][:3, 2]
            o_i = T[i][:3, 3]

            J[:3, i] = np.cross(z_i, o_ee - o_i)
            J[3:, i] = z_i

        return J

    def check_singularity(self, theta):
        J = self.jacobian(theta)
        J_pos = J[:3, :]

        JJT = J_pos @ J_pos.T
        det_JJT = np.linalg.det(JJT)

        try:
            condition_number = np.linalg.cond(J_pos)
        except:
            condition_number = np.inf

        manipulability = np.sqrt(abs(det_JJT))

        det_threshold = 1e-4
        cond_threshold = 1e4
        manip_threshold = 1e-3

        is_singular = (abs(det_JJT) < det_threshold or
                       condition_number > cond_threshold or
                       manipulability < manip_threshold)

        return is_singular, manipulability, condition_number

    def inverse_kinematics(self, target_pos, max_iter=100, tol=1e-4):
        theta = self.theta.copy()
        singularity_detected = False

        for iteration in range(max_iter):
            T_current = self.forward_kinematics(theta)
            current_pos = T_current[:3, 3]
            error = target_pos - current_pos

            if np.linalg.norm(error) < tol:
                is_singular, manip, cond = self.check_singularity(theta)
                return theta, True, iteration, is_singular, manip, cond

            J = self.jacobian(theta)
            J_pos = J[:3, :]

            is_singular, manip, cond = self.check_singularity(theta)
            if is_singular:
                singularity_detected = True

            lambda_damping = 0.01
            try:
                delta_theta = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + lambda_damping ** 2 * np.eye(3)) @ error
            except np.linalg.LinAlgError:
                is_singular, manip, cond = self.check_singularity(theta)
                return theta, False, iteration, True, manip, cond

            theta += delta_theta
            theta = np.clip(theta, -2 * np.pi, 2 * np.pi)

        is_singular, manip, cond = self.check_singularity(theta)
        return theta, False, max_iter, singularity_detected or is_singular, manip, cond

    def move_line(self, target_pos, num_steps=20):
        start_pos = self.current_pos
        trajectory = []
        trajectory_positions = []
        singularity_flags = []

        for i in range(num_steps + 1):
            t = i / num_steps
            intermediate_pos = start_pos + t * (target_pos - start_pos)

            theta_new, success, _, is_singular, _, _ = self.inverse_kinematics(intermediate_pos)
            if success:
                trajectory.append(theta_new)
                T_ee = self.forward_kinematics(theta_new)
                trajectory_positions.append(T_ee[:3, 3])
                singularity_flags.append(is_singular)
            else:
                if len(trajectory) > 0:
                    break

        return trajectory, np.array(trajectory_positions), singularity_flags

    def move_square(self, size=0.1, plane='xy', num_steps_per_side=15):
        start_pos = self.current_pos

        if plane == 'xy':
            axis1 = np.array([1, 0, 0])
            axis2 = np.array([0, 1, 0])
        elif plane == 'xz':
            axis1 = np.array([1, 0, 0])
            axis2 = np.array([0, 0, 1])
        else:
            axis1 = np.array([0, 1, 0])
            axis2 = np.array([0, 0, 1])

        corner1 = start_pos
        corner2 = start_pos + size * axis1
        corner3 = start_pos + size * axis1 + size * axis2
        corner4 = start_pos + size * axis2

        waypoints = [corner1, corner2, corner3, corner4, corner1]

        trajectory = []
        trajectory_positions = []
        singularity_flags = []

        for i in range(len(waypoints) - 1):
            start_wp = waypoints[i]
            end_wp = waypoints[i + 1]

            for j in range(num_steps_per_side + 1):
                if i == len(waypoints) - 2 and j == num_steps_per_side:
                    break

                t = j / num_steps_per_side
                intermediate_pos = start_wp + t * (end_wp - start_wp)

                theta_new, success, _, is_singular, _, _ = self.inverse_kinematics(intermediate_pos)
                if success:
                    trajectory.append(theta_new)
                    T_ee = self.forward_kinematics(theta_new)
                    trajectory_positions.append(T_ee[:3, 3])
                    singularity_flags.append(is_singular)

        return trajectory, np.array(trajectory_positions), singularity_flags

    def move_circle(self, radius=0.1, plane='xy', num_steps=60):
        start_pos = self.current_pos

        if plane == 'xy':
            axis1 = np.array([1, 0, 0])
            axis2 = np.array([0, 1, 0])
        elif plane == 'xz':
            axis1 = np.array([1, 0, 0])
            axis2 = np.array([0, 0, 1])
        else:
            axis1 = np.array([0, 1, 0])
            axis2 = np.array([0, 0, 1])

        center = start_pos + radius * axis1

        trajectory = []
        trajectory_positions = []
        singularity_flags = []

        for i in range(num_steps + 1):
            angle = 2 * np.pi * i / num_steps
            circle_pos = center + radius * (np.cos(angle + np.pi) * axis1 + np.sin(angle + np.pi) * axis2)

            theta_new, success, _, is_singular, _, _ = self.inverse_kinematics(circle_pos)
            if success:
                trajectory.append(theta_new)
                T_ee = self.forward_kinematics(theta_new)
                trajectory_positions.append(T_ee[:3, 3])
                singularity_flags.append(is_singular)

        return trajectory, np.array(trajectory_positions), singularity_flags

    def reset_to_home(self):
        self.theta = self.home_theta.copy()
        self.current_pos = self.forward_kinematics(self.theta)[:3, 3]


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

        self.traj_sliders = []
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

        if is_singular:
            self.status_text = "WARNING: Singularity Detected!"
        else:
            self.status_text = "Forward Kinematics Mode"

        self.update_plot()

    def execute_trajectory(self, event):
        if self.animating:
            return

        target_pos = np.array([
            self.traj_sliders[0].val,
            self.traj_sliders[1].val,
            self.traj_sliders[2].val
        ])

        if self.traj_type == 'Line':
            self.trajectory, self.trajectory_positions, self.singularity_flags = self.robot.move_line(target_pos)
        elif self.traj_type == 'Square':
            traj_line, _, _ = self.robot.move_line(target_pos)
            if traj_line:
                self.robot.theta = traj_line[-1]
                self.robot.current_pos = self.robot.forward_kinematics(self.robot.theta)[:3, 3]
            self.trajectory, self.trajectory_positions, self.singularity_flags = self.robot.move_square(size=0.1,
                                                                                                        plane=self.plane)
        else:
            traj_line, _, _ = self.robot.move_line(target_pos)
            if traj_line:
                self.robot.theta = traj_line[-1]
                self.robot.current_pos = self.robot.forward_kinematics(self.robot.theta)[:3, 3]
            self.trajectory, self.trajectory_positions, self.singularity_flags = self.robot.move_circle(radius=0.1,
                                                                                                        plane=self.plane)

        if self.trajectory:
            self.traj_index = 0
            self.animating = True
            self.trajectory_blocked = False
            self.animate()

    def draw_coordinate_frame(self, T, scale=0.08):
        origin = T[:3, 3]
        x_axis = T[:3, 0] * scale
        y_axis = T[:3, 1] * scale
        z_axis = T[:3, 2] * scale

        x_end = origin + x_axis
        self.ax.plot([origin[0], x_end[0]], [origin[1], x_end[1]], [origin[2], x_end[2]], 'r-', linewidth=4)

        y_end = origin + y_axis
        self.ax.plot([origin[0], y_end[0]], [origin[1], y_end[1]], [origin[2], y_end[2]], 'b-', linewidth=4)

        z_end = origin + z_axis
        self.ax.plot([origin[0], z_end[0]], [origin[1], z_end[1]], [origin[2], z_end[2]], 'g-', linewidth=4)

    def draw_end_effector(self, T_ee):
        ee_pos = T_ee[:3, 3]
        self.ax.scatter([ee_pos[0]], [ee_pos[1]], [ee_pos[2]],
                        c='orange', s=500, marker='*', edgecolors='black', linewidths=4, zorder=10)

    def update_info_display(self):
        self.ax_info.clear()
        self.ax_info.axis('off')

        bg_color = 'lightsalmon' if self.is_singular else 'lightblue'

        T_ee = self.robot.forward_kinematics(self.robot.theta)

        header = "WARNING: SINGULARITY DETECTED\n" if self.is_singular else ""
        info_text = header
        info_text += f"Mode: {self.control_mode}\n"
        info_text += "=" * 40 + "\n\n"

        info_text += "Joint Angles:\n"
        for i in range(6):
            angle_deg = np.degrees(self.robot.theta[i])
            info_text += f"  θ{i + 1}: {angle_deg:>8.2f}°\n"

        info_text += "\n" + "=" * 40 + "\n"
        info_text += "End Effector Transform:\n"
        for i in range(4):
            info_text += "["
            for j in range(4):
                info_text += f"{T_ee[i, j]:>7.3f}"
                if j < 3:
                    info_text += " "
            info_text += "]\n"

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

        info_text += "\n"
        info_text += f"Status: {self.status_text}\n"

        self.ax_info.text(0.0, 0.5, info_text,
                          fontsize=15,
                          verticalalignment='center',
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
        self.traj_sliders[0].set_val(home_pos[0])
        self.traj_sliders[1].set_val(home_pos[1])
        self.traj_sliders[2].set_val(home_pos[2])

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

                _, manip, cond = self.robot.check_singularity(self.robot.theta)
                self.manipulability = manip
                self.condition_number = cond

                self.status_text = "WARNING: Trajectory stopped at singularity!"
                self.update_plot()
                return

            self.robot.theta = self.trajectory[self.traj_index]
            self.robot.current_pos = self.robot.forward_kinematics(self.robot.theta)[:3, 3]

            is_singular, manip, cond = self.robot.check_singularity(self.robot.theta)
            self.is_singular = is_singular
            self.manipulability = manip
            self.condition_number = cond

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


if __name__ == "__main__":
    visualizer = UR5Visualizer()