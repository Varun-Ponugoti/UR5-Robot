import numpy as np


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
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0,        sa,       ca,      d],
            [0,         0,        0,      1]
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

        is_singular = (abs(det_JJT) < 1e-4 or
                       condition_number > 1e4 or
                       manipulability < 1e-3)

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
        trajectory, trajectory_positions, singularity_flags = [], [], []

        for i in range(num_steps + 1):
            t = i / num_steps
            intermediate_pos = start_pos + t * (target_pos - start_pos)
            theta_new, success, _, is_singular, _, _ = self.inverse_kinematics(intermediate_pos)
            if success:
                trajectory.append(theta_new)
                trajectory_positions.append(self.forward_kinematics(theta_new)[:3, 3])
                singularity_flags.append(is_singular)
            else:
                if len(trajectory) > 0:
                    break

        return trajectory, np.array(trajectory_positions), singularity_flags

    def move_square(self, size=0.1, plane='xy', num_steps_per_side=15):
        start_pos = self.current_pos

        if plane == 'xy':
            axis1, axis2 = np.array([1, 0, 0]), np.array([0, 1, 0])
        elif plane == 'xz':
            axis1, axis2 = np.array([1, 0, 0]), np.array([0, 0, 1])
        else:
            axis1, axis2 = np.array([0, 1, 0]), np.array([0, 0, 1])

        waypoints = [
            start_pos,
            start_pos + size * axis1,
            start_pos + size * axis1 + size * axis2,
            start_pos + size * axis2,
            start_pos,
        ]

        trajectory, trajectory_positions, singularity_flags = [], [], []

        for i in range(len(waypoints) - 1):
            for j in range(num_steps_per_side + 1):
                if i == len(waypoints) - 2 and j == num_steps_per_side:
                    break
                t = j / num_steps_per_side
                intermediate_pos = waypoints[i] + t * (waypoints[i + 1] - waypoints[i])
                theta_new, success, _, is_singular, _, _ = self.inverse_kinematics(intermediate_pos)
                if success:
                    trajectory.append(theta_new)
                    trajectory_positions.append(self.forward_kinematics(theta_new)[:3, 3])
                    singularity_flags.append(is_singular)

        return trajectory, np.array(trajectory_positions), singularity_flags

    def move_circle(self, radius=0.1, plane='xy', num_steps=60):
        start_pos = self.current_pos

        if plane == 'xy':
            axis1, axis2 = np.array([1, 0, 0]), np.array([0, 1, 0])
        elif plane == 'xz':
            axis1, axis2 = np.array([1, 0, 0]), np.array([0, 0, 1])
        else:
            axis1, axis2 = np.array([0, 1, 0]), np.array([0, 0, 1])

        center = start_pos + radius * axis1
        trajectory, trajectory_positions, singularity_flags = [], [], []

        for i in range(num_steps + 1):
            angle = 2 * np.pi * i / num_steps
            circle_pos = center + radius * (np.cos(angle + np.pi) * axis1 + np.sin(angle + np.pi) * axis2)
            theta_new, success, _, is_singular, _, _ = self.inverse_kinematics(circle_pos)
            if success:
                trajectory.append(theta_new)
                trajectory_positions.append(self.forward_kinematics(theta_new)[:3, 3])
                singularity_flags.append(is_singular)

        return trajectory, np.array(trajectory_positions), singularity_flags

    def reset_to_home(self):
        self.theta = self.home_theta.copy()
        self.current_pos = self.forward_kinematics(self.theta)[:3, 3]
