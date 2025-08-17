import numpy as np
import pybullet as p
from typing import List, Tuple, Optional
from scipy.optimize import minimize
import math

class RM65Controller:
    """RM65机械臂控制器，提供运动学和动力学控制功能"""
    
    def __init__(self, env):
        """
        初始化控制器
        
        Args:
            env: RM65Environment实例
        """
        self.env = env
        self.robot_id = env.robot_id
        self.joint_indices = env.joint_indices
        self.joint_limits = env.joint_limits
        self.num_joints = len(self.joint_indices)
        
        # 控制参数
        self.position_gain = 0.1
        self.velocity_gain = 0.01
        self.max_velocity = 2.0
        self.max_acceleration = 5.0
        
        # 末端执行器链接索引（通常是最后一个关节）
        self.end_effector_link = self.num_joints - 1
        
    def forward_kinematics(self, joint_positions: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        正运动学：根据关节角度计算末端执行器位姿
        
        Args:
            joint_positions: 关节角度列表
            
        Returns:
            position: 末端执行器位置 [x, y, z]
            orientation: 末端执行器姿态四元数 [x, y, z, w]
        """
        # 临时设置关节位置
        original_positions, _ = self.env.get_joint_states()
        
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx, joint_positions[i])
            
        # 获取末端执行器位姿
        link_state = p.getLinkState(self.robot_id, self.end_effector_link)
        position = np.array(link_state[0])
        orientation = np.array(link_state[1])
        
        # 恢复原始关节位置
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx, original_positions[i])
            
        return position, orientation
        
    def jacobian(self, joint_positions: List[float]) -> np.ndarray:
        """
        计算雅可比矩阵
        
        Args:
            joint_positions: 关节角度列表
            
        Returns:
            jacobian: 6xN雅可比矩阵 (位置3维 + 姿态3维)
        """
        # 临时设置关节位置
        original_positions, _ = self.env.get_joint_states()
        
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx, joint_positions[i])
            
        # 计算雅可比矩阵
        zero_vec = [0.0] * self.num_joints
        jac_t, jac_r = p.calculateJacobian(
            self.robot_id,
            self.end_effector_link,
            [0, 0, 0],  # 局部位置
            joint_positions,
            zero_vec,
            zero_vec
        )
        
        # 恢复原始关节位置
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx, original_positions[i])
            
        # 组合平移和旋转雅可比
        jacobian = np.vstack([np.array(jac_t), np.array(jac_r)])
        return jacobian
        
    def inverse_kinematics(self, target_position: List[float], 
                          target_orientation: Optional[List[float]] = None,
                          initial_guess: Optional[List[float]] = None) -> List[float]:
        """
        逆运动学：根据目标位姿计算关节角度
        
        Args:
            target_position: 目标位置 [x, y, z]
            target_orientation: 目标姿态四元数 [x, y, z, w]，可选
            initial_guess: 初始猜测关节角度，可选
            
        Returns:
            joint_positions: 计算得到的关节角度
        """
        if initial_guess is None:
            initial_guess = [0.0] * self.num_joints
            
        if target_orientation is None:
            # 只考虑位置的逆运动学
            result = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_link,
                target_position,
                lowerLimits=[limit[0] for limit in self.joint_limits],
                upperLimits=[limit[1] for limit in self.joint_limits],
                jointRanges=[limit[1] - limit[0] for limit in self.joint_limits],
                restPoses=initial_guess
            )
        else:
            # 考虑位置和姿态的逆运动学
            result = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_link,
                target_position,
                target_orientation,
                lowerLimits=[limit[0] for limit in self.joint_limits],
                upperLimits=[limit[1] for limit in self.joint_limits],
                jointRanges=[limit[1] - limit[0] for limit in self.joint_limits],
                restPoses=initial_guess
            )
            
        return list(result[:self.num_joints])
        
    def velocity_ik(self, target_velocity: np.ndarray, 
                   current_positions: List[float]) -> List[float]:
        """
        速度层逆运动学
        
        Args:
            target_velocity: 目标末端执行器速度 [vx, vy, vz, wx, wy, wz]
            current_positions: 当前关节位置
            
        Returns:
            joint_velocities: 关节速度
        """
        # 计算雅可比矩阵
        J = self.jacobian(current_positions)
        
        # 使用伪逆求解
        J_pinv = np.linalg.pinv(J)
        joint_velocities = J_pinv @ target_velocity
        
        # 限制关节速度
        joint_velocities = np.clip(joint_velocities, -self.max_velocity, self.max_velocity)
        
        return joint_velocities.tolist()
        
    def cartesian_impedance_control(self, target_position: np.ndarray,
                                   target_orientation: np.ndarray,
                                   current_positions: List[float],
                                   current_velocities: List[float],
                                   stiffness: float = 1000.0,
                                   damping: float = 50.0) -> List[float]:
        """
        笛卡尔阻抗控制
        
        Args:
            target_position: 目标位置
            target_orientation: 目标姿态
            current_positions: 当前关节位置
            current_velocities: 当前关节速度
            stiffness: 刚度系数
            damping: 阻尼系数
            
        Returns:
            joint_torques: 关节力矩
        """
        # 获取当前末端执行器位姿
        current_pos, current_ori = self.forward_kinematics(current_positions)
        
        # 位置误差
        pos_error = target_position - current_pos
        
        # 姿态误差（简化处理）
        ori_error = np.array([0, 0, 0])  # 这里可以实现更复杂的姿态误差计算
        
        # 笛卡尔空间力
        cartesian_force = stiffness * np.concatenate([pos_error, ori_error])
        
        # 计算雅可比矩阵
        J = self.jacobian(current_positions)
        
        # 转换为关节空间力矩
        joint_torques = J.T @ cartesian_force
        
        # 添加阻尼
        joint_torques -= damping * np.array(current_velocities)
        
        return joint_torques.tolist()
        
    def trajectory_planning(self, start_pos: List[float], end_pos: List[float],
                          duration: float, num_points: int = 100) -> List[List[float]]:
        """
        关节空间轨迹规划（五次多项式）
        
        Args:
            start_pos: 起始关节位置
            end_pos: 结束关节位置
            duration: 运动时间
            num_points: 轨迹点数量
            
        Returns:
            trajectory: 轨迹点列表，每个点包含关节位置
        """
        trajectory = []
        
        for i in range(num_points):
            t = i * duration / (num_points - 1)
            s = self._quintic_polynomial(t, duration)
            
            # 插值计算当前位置
            current_pos = []
            for j in range(self.num_joints):
                pos = start_pos[j] + s * (end_pos[j] - start_pos[j])
                current_pos.append(pos)
                
            trajectory.append(current_pos)
            
        return trajectory
        
    def cartesian_trajectory_planning(self, start_pose: Tuple[np.ndarray, np.ndarray],
                                    end_pose: Tuple[np.ndarray, np.ndarray],
                                    duration: float, num_points: int = 100) -> List[List[float]]:
        """
        笛卡尔空间轨迹规划
        
        Args:
            start_pose: 起始位姿 (position, orientation)
            end_pose: 结束位姿 (position, orientation)
            duration: 运动时间
            num_points: 轨迹点数量
            
        Returns:
            trajectory: 关节空间轨迹点列表
        """
        trajectory = []
        start_pos, start_ori = start_pose
        end_pos, end_ori = end_pose
        
        # 获取起始关节位置
        current_joint_pos, _ = self.env.get_joint_states()
        
        for i in range(num_points):
            t = i * duration / (num_points - 1)
            s = self._quintic_polynomial(t, duration)
            
            # 位置插值
            target_pos = start_pos + s * (end_pos - start_pos)
            
            # 姿态插值（简化为线性插值）
            target_ori = start_ori + s * (end_ori - start_ori)
            target_ori = target_ori / np.linalg.norm(target_ori)  # 归一化
            
            # 逆运动学求解
            joint_pos = self.inverse_kinematics(
                target_pos.tolist(), 
                target_ori.tolist(),
                current_joint_pos
            )
            
            trajectory.append(joint_pos)
            current_joint_pos = joint_pos
            
        return trajectory
        
    def _quintic_polynomial(self, t: float, T: float) -> float:
        """
        五次多项式轨迹规划
        
        Args:
            t: 当前时间
            T: 总时间
            
        Returns:
            s: 归一化位置 (0到1)
        """
        if t <= 0:
            return 0.0
        elif t >= T:
            return 1.0
        else:
            tau = t / T
            return 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            
    def check_joint_limits(self, joint_positions: List[float]) -> bool:
        """
        检查关节位置是否在限制范围内
        
        Args:
            joint_positions: 关节位置列表
            
        Returns:
            valid: 是否在限制范围内
        """
        for i, pos in enumerate(joint_positions):
            if i < len(self.joint_limits):
                lower, upper = self.joint_limits[i]
                if pos < lower or pos > upper:
                    return False
        return True
        
    def clamp_joint_positions(self, joint_positions: List[float]) -> List[float]:
        """
        将关节位置限制在允许范围内
        
        Args:
            joint_positions: 关节位置列表
            
        Returns:
            clamped_positions: 限制后的关节位置
        """
        clamped = []
        for i, pos in enumerate(joint_positions):
            if i < len(self.joint_limits):
                lower, upper = self.joint_limits[i]
                clamped.append(np.clip(pos, lower, upper))
            else:
                clamped.append(pos)
        return clamped
        
    def get_workspace_bounds(self, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        估算机械臂工作空间边界
        
        Args:
            num_samples: 采样点数量
            
        Returns:
            min_bounds: 最小边界 [x_min, y_min, z_min]
            max_bounds: 最大边界 [x_max, y_max, z_max]
        """
        positions = []
        
        for _ in range(num_samples):
            # 随机生成关节角度
            joint_pos = []
            for limit in self.joint_limits:
                angle = np.random.uniform(limit[0], limit[1])
                joint_pos.append(angle)
                
            # 计算末端执行器位置
            pos, _ = self.forward_kinematics(joint_pos)
            positions.append(pos)
            
        positions = np.array(positions)
        min_bounds = np.min(positions, axis=0)
        max_bounds = np.max(positions, axis=0)
        
        return min_bounds, max_bounds


if __name__ == "__main__":
    # 测试控制器
    from rm65_env import RM65Environment
    import time
    
    env = RM65Environment(gui=True)
    controller = RM65Controller(env)
    
    try:
        # 测试正运动学
        joint_pos = [0.0] * controller.num_joints
        pos, ori = controller.forward_kinematics(joint_pos)
        print(f"末端执行器位置: {pos}")
        print(f"末端执行器姿态: {ori}")
        
        # 测试逆运动学
        target_pos = [0.3, 0.3, 0.5]
        ik_result = controller.inverse_kinematics(target_pos)
        print(f"逆运动学结果: {ik_result}")
        
        # 测试轨迹规划
        start_pos = [0.0] * controller.num_joints
        end_pos = ik_result
        trajectory = controller.trajectory_planning(start_pos, end_pos, 3.0, 100)
        
        # 执行轨迹
        for point in trajectory:
            env.set_joint_positions(point)
            env.step_simulation()
            time.sleep(0.03)
            
    except KeyboardInterrupt:
        print("测试中断")
    finally:
        env.close()