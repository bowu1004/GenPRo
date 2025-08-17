import gym
from gym import spaces
import numpy as np
import pybullet as p
import random
import math
from typing import Tuple, Dict, Any, Optional
from rm65_env import RM65Environment
from rm65_controller import RM65Controller

class RM65ReachEnv(gym.Env):
    """RM65机械臂到达目标点的强化学习环境"""
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 gui=False,
                 max_episode_steps=200,
                 target_range=0.8,
                 action_scale=0.1,
                 use_vision=True,
                 image_size=(224, 224)):
        """
        初始化强化学习环境
        
        Args:
            gui: 是否显示GUI
            max_episode_steps: 最大步数
            target_range: 目标点生成范围
            action_scale: 动作缩放因子
            use_vision: 是否使用视觉观察
            image_size: 图像尺寸
        """
        super(RM65ReachEnv, self).__init__()
        
        # 环境参数
        self.max_episode_steps = max_episode_steps
        self.target_range = target_range
        self.action_scale = action_scale
        self.use_vision = use_vision
        self.image_size = image_size
        
        # 初始化仿真环境
        self.env = RM65Environment(gui=gui)
        self.controller = RM65Controller(self.env)
        
        # 环境状态
        self.current_step = 0
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.target_sphere_id = None
        
        # 定义动作空间（关节角度增量）
        self.num_joints = len(self.controller.joint_indices)
        action_low = np.array([-1.0] * self.num_joints, dtype=np.float32)
        action_high = np.array([1.0] * self.num_joints, dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        
        # 定义观察空间
        self._setup_observation_space()
        
        # 奖励参数
        self.distance_threshold = 0.05  # 成功距离阈值
        self.previous_distance = 0.0
        
        # 工作空间边界
        self.workspace_center = np.array([0.0, 0.0, 0.5])
        self.workspace_radius = 0.6
        
    def _setup_observation_space(self):
        """设置观察空间"""
        # 关节状态维度：位置 + 速度
        joint_dim = self.num_joints * 2
        
        # 末端执行器状态维度：位置 + 姿态
        ee_dim = 7  # 3位置 + 4四元数
        
        # 目标位置维度
        target_dim = 3
        
        # 其他状态维度
        other_dim = 2  # 距离 + 步数归一化
        
        state_dim = joint_dim + ee_dim + target_dim + other_dim
        
        if self.use_vision:
            # 视觉观察 + 状态向量
            self.observation_space = spaces.Dict({
                'image': spaces.Box(
                    low=0, high=255, 
                    shape=(self.image_size[1], self.image_size[0], 3), 
                    dtype=np.uint8
                ),
                'state': spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(state_dim,), 
                    dtype=np.float32
                )
            })
        else:
            # 仅状态向量
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(state_dim,), 
                dtype=np.float32
            )
            
    def reset(self) -> Dict[str, Any]:
        """重置环境"""
        self.current_step = 0
        
        # 重置机械臂到随机初始位置
        initial_positions = []
        for limit in self.controller.joint_limits:
            pos = np.random.uniform(limit[0] * 0.5, limit[1] * 0.5)
            initial_positions.append(pos)
        
        self.env.reset_robot(initial_positions)
        
        # 生成随机目标位置
        self._generate_target()
        
        # 获取初始观察
        observation = self._get_observation()
        
        # 计算初始距离
        ee_pos, _ = self.env.get_end_effector_pose()
        self.previous_distance = np.linalg.norm(np.array(ee_pos) - self.target_position)
        
        return observation
        
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """执行一步动作"""
        self.current_step += 1
        
        # 获取当前关节状态
        current_positions, current_velocities = self.env.get_joint_states()
        
        # 应用动作（关节角度增量）
        action = np.clip(action, -1.0, 1.0)
        target_positions = np.array(current_positions) + action * self.action_scale
        
        # 限制关节角度在允许范围内
        target_positions = self.controller.clamp_joint_positions(target_positions.tolist())
        
        # 设置关节目标位置
        self.env.set_joint_positions(target_positions)
        
        # 执行仿真步骤
        for _ in range(5):  # 每个RL步骤执行5个仿真步骤
            self.env.step_simulation()
            
        # 获取新的观察
        observation = self._get_observation()
        
        # 计算奖励
        reward = self._compute_reward()
        
        # 检查是否结束
        done = self._is_done()
        
        # 信息字典
        info = self._get_info()
        
        return observation, reward, done, info
        
    def _generate_target(self):
        """生成随机目标位置"""
        # 在工作空间内生成随机目标
        while True:
            # 球坐标系生成
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            r = np.random.uniform(0.2, self.target_range)
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi) + 0.3  # 抬高基准高度
            
            self.target_position = np.array([x, y, z])
            
            # 检查是否在工作空间内
            if self._is_position_reachable(self.target_position):
                break
                
        # 移除旧的目标球体
        if self.target_sphere_id is not None:
            p.removeBody(self.target_sphere_id)
            
        # 创建目标可视化球体
        self.target_sphere_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_SPHERE, radius=0.03
            ),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 0.8]
            ),
            basePosition=self.target_position
        )
        
    def _is_position_reachable(self, position: np.ndarray) -> bool:
        """检查位置是否可达"""
        # 简单的工作空间检查
        distance_from_base = np.linalg.norm(position[:2])  # 水平距离
        height = position[2]
        
        return (distance_from_base <= self.workspace_radius and 
                0.1 <= height <= 1.0)
                
    def _get_observation(self) -> Dict[str, Any]:
        """获取当前观察"""
        # 获取关节状态
        joint_positions, joint_velocities = self.env.get_joint_states()
        
        # 获取末端执行器状态
        ee_position, ee_orientation = self.env.get_end_effector_pose()
        
        # 计算到目标的距离
        distance_to_target = np.linalg.norm(np.array(ee_position) - self.target_position)
        
        # 归一化步数
        normalized_step = self.current_step / self.max_episode_steps
        
        # 组合状态向量
        state_vector = np.concatenate([
            joint_positions,
            joint_velocities,
            ee_position,
            ee_orientation,
            self.target_position,
            [distance_to_target, normalized_step]
        ]).astype(np.float32)
        
        if self.use_vision:
            # 获取相机图像
            image = self.env.get_camera_image(
                width=self.image_size[0], 
                height=self.image_size[1]
            )
            
            return {
                'image': image.astype(np.uint8),
                'state': state_vector
            }
        else:
            return state_vector
            
    def _compute_reward(self) -> float:
        """计算奖励函数"""
        # 获取当前末端执行器位置
        ee_position, _ = self.env.get_end_effector_pose()
        current_distance = np.linalg.norm(np.array(ee_position) - self.target_position)
        
        # 距离奖励（越近越好）
        distance_reward = -current_distance
        
        # 进步奖励（距离减少）
        progress_reward = (self.previous_distance - current_distance) * 10.0
        self.previous_distance = current_distance
        
        # 成功奖励
        success_reward = 0.0
        if current_distance < self.distance_threshold:
            success_reward = 100.0
            
        # 时间惩罚
        time_penalty = -0.1
        
        # 关节限制惩罚
        joint_positions, _ = self.env.get_joint_states()
        joint_penalty = 0.0
        for i, pos in enumerate(joint_positions):
            if i < len(self.controller.joint_limits):
                lower, upper = self.controller.joint_limits[i]
                if pos <= lower + 0.1 or pos >= upper - 0.1:
                    joint_penalty -= 1.0
                    
        # 总奖励
        total_reward = (distance_reward + progress_reward + 
                       success_reward + time_penalty + joint_penalty)
        
        return total_reward
        
    def _is_done(self) -> bool:
        """检查是否结束"""
        # 达到最大步数
        if self.current_step >= self.max_episode_steps:
            return True
            
        # 成功到达目标
        ee_position, _ = self.env.get_end_effector_pose()
        distance = np.linalg.norm(np.array(ee_position) - self.target_position)
        if distance < self.distance_threshold:
            return True
            
        # 末端执行器超出工作空间
        if not self._is_position_reachable(np.array(ee_position)):
            return True
            
        return False
        
    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        ee_position, _ = self.env.get_end_effector_pose()
        distance = np.linalg.norm(np.array(ee_position) - self.target_position)
        
        return {
            'distance_to_target': distance,
            'is_success': distance < self.distance_threshold,
            'current_step': self.current_step,
            'target_position': self.target_position.copy(),
            'ee_position': np.array(ee_position)
        }
        
    def render(self, mode='human'):
        """渲染环境"""
        if mode == 'rgb_array':
            return self.env.get_camera_image()
        elif mode == 'human':
            # GUI模式下自动渲染
            pass
            
    def close(self):
        """关闭环境"""
        if self.target_sphere_id is not None:
            p.removeBody(self.target_sphere_id)
        self.env.close()
        
    def seed(self, seed=None):
        """设置随机种子"""
        np.random.seed(seed)
        random.seed(seed)
        return [seed]


class RM65ManipulationEnv(RM65ReachEnv):
    """RM65机械臂操作环境（抓取任务）"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 添加物体
        self.object_id = None
        self.object_initial_pos = [0.3, 0.3, 0.1]
        
    def reset(self):
        """重置环境并添加操作物体"""
        observation = super().reset()
        
        # 移除旧物体
        if self.object_id is not None:
            p.removeBody(self.object_id)
            
        # 创建操作物体（立方体）
        self.object_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02]
            ),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], 
                rgbaColor=[0, 1, 0, 1]
            ),
            basePosition=self.object_initial_pos
        )
        
        return observation
        
    def _compute_reward(self):
        """计算操作任务的奖励"""
        # 基础到达奖励
        reach_reward = super()._compute_reward()
        
        # 物体位置奖励
        object_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        object_to_target = np.linalg.norm(np.array(object_pos) - self.target_position)
        
        manipulation_reward = -object_to_target * 5.0
        
        # 如果物体接近目标，给予额外奖励
        if object_to_target < 0.1:
            manipulation_reward += 50.0
            
        return reach_reward + manipulation_reward
        
    def close(self):
        """关闭环境"""
        if self.object_id is not None:
            p.removeBody(self.object_id)
        super().close()


if __name__ == "__main__":
    # 测试环境
    env = RM65ReachEnv(gui=True, use_vision=False)
    
    try:
        obs = env.reset()
        print(f"观察空间: {env.observation_space}")
        print(f"动作空间: {env.action_space}")
        print(f"初始观察形状: {obs.shape if isinstance(obs, np.ndarray) else {k: v.shape for k, v in obs.items()}}")
        
        for episode in range(3):
            obs = env.reset()
            total_reward = 0
            
            for step in range(200):
                # 随机动作
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                total_reward += reward
                
                if done:
                    print(f"Episode {episode}: 步数={step+1}, 总奖励={total_reward:.2f}, 成功={info['is_success']}")
                    break
                    
    except KeyboardInterrupt:
        print("测试中断")
    finally:
        env.close()