import pybullet as p
import pybullet_data
import numpy as np
import time
import os
from typing import List, Tuple, Optional

class RM65Environment:
    """RM65机械臂PyBullet仿真环境"""
    
    def __init__(self, gui=True, urdf_path=None):
        """
        初始化RM65仿真环境
        
        Args:
            gui: 是否显示GUI界面
            urdf_path: URDF文件路径
        """
        self.gui = gui
        self.physics_client = None
        self.robot_id = None
        self.num_joints = 0
        self.joint_indices = []
        self.joint_limits = []
        self.joint_ranges = []
        
        # 默认URDF路径
        if urdf_path is None:
            self.urdf_path = os.path.join(
                os.path.dirname(__file__), 
                "..", "genesis", "assets", "RM65", "urdf", 
                "rm_65_b_description", "urdf", "rm_65_b_description.urdf"
            )
        else:
            self.urdf_path = urdf_path
            
        self.initialize_simulation()
        
    def initialize_simulation(self):
        """初始化PyBullet仿真环境"""
        # 连接物理引擎
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        # 设置重力
        p.setGravity(0, 0, -9.81)
        
        # 添加搜索路径
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 加载地面
        self.plane_id = p.loadURDF("plane.urdf")
        
        # 加载机械臂
        self.load_robot()
        
        # 设置相机
        self.setup_camera()
        
    def load_robot(self):
        """加载RM65机械臂模型"""
        try:
            # 加载URDF文件
            start_pos = [0, 0, 0]
            start_orientation = p.getQuaternionFromEuler([0, 0, 0])
            
            self.robot_id = p.loadURDF(
                self.urdf_path,
                start_pos,
                start_orientation,
                useFixedBase=True
            )
            
            # 获取关节信息
            self.num_joints = p.getNumJoints(self.robot_id)
            print(f"机械臂关节数量: {self.num_joints}")
            
            # 分析关节信息
            self.analyze_joints()
            
        except Exception as e:
            print(f"加载机械臂失败: {e}")
            raise
            
    def analyze_joints(self):
        """分析机械臂关节信息"""
        self.joint_indices = []
        self.joint_limits = []
        self.joint_ranges = []
        
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            # 只考虑旋转关节
            if joint_type == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                self.joint_limits.append([lower_limit, upper_limit])
                self.joint_ranges.append(upper_limit - lower_limit)
                
                print(f"关节 {i}: {joint_name}, 类型: {joint_type}, 范围: [{lower_limit:.3f}, {upper_limit:.3f}]")
                
    def setup_camera(self):
        """设置相机参数"""
        self.camera_distance = 1.5
        self.camera_yaw = 45
        self.camera_pitch = -30
        self.camera_target = [0, 0, 0.5]
        
        p.resetDebugVisualizerCamera(
            self.camera_distance,
            self.camera_yaw,
            self.camera_pitch,
            self.camera_target
        )
        
    def reset_robot(self, joint_positions=None):
        """重置机械臂到指定位置"""
        if joint_positions is None:
            # 默认位置（所有关节为0）
            joint_positions = [0.0] * len(self.joint_indices)
            
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx, joint_positions[i])
            
    def get_joint_states(self) -> Tuple[List[float], List[float]]:
        """获取关节状态"""
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        positions = [state[0] for state in joint_states]
        velocities = [state[1] for state in joint_states]
        return positions, velocities
        
    def set_joint_positions(self, positions: List[float]):
        """设置关节位置"""
        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=positions[i]
            )
            
    def set_joint_velocities(self, velocities: List[float]):
        """设置关节速度"""
        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.VELOCITY_CONTROL,
                targetVelocity=velocities[i]
            )
            
    def set_joint_torques(self, torques: List[float]):
        """设置关节力矩"""
        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.TORQUE_CONTROL,
                force=torques[i]
            )
            
    def get_end_effector_pose(self) -> Tuple[List[float], List[float]]:
        """获取末端执行器位姿"""
        # 假设末端执行器是最后一个link
        end_effector_link = self.num_joints - 1
        link_state = p.getLinkState(self.robot_id, end_effector_link)
        position = list(link_state[0])
        orientation = list(link_state[1])
        return position, orientation
        
    def get_camera_image(self, width=224, height=224) -> np.ndarray:
        """获取相机图像"""
        # 计算视图矩阵和投影矩阵
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=0,
            upAxisIndex=2
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width/height,
            nearVal=0.1,
            farVal=100.0
        )
        
        # 渲染图像
        _, _, rgb_array, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix
        )
        
        # 转换为numpy数组
        rgb_array = np.array(rgb_array).reshape(height, width, 4)[:, :, :3]
        return rgb_array
        
    def step_simulation(self):
        """执行一步仿真"""
        p.stepSimulation()
        
    def close(self):
        """关闭仿真环境"""
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except p.error:
                # 如果已经断开连接，忽略错误
                pass
            self.physics_client = None
            
    def __del__(self):
        """析构函数"""
        self.close()


if __name__ == "__main__":
    # 测试环境
    env = RM65Environment(gui=True)
    
    try:
        # 重置机械臂
        env.reset_robot()
        
        # 运行仿真
        for i in range(1000):
            # 获取当前状态
            positions, velocities = env.get_joint_states()
            
            # 简单的正弦波运动
            target_positions = [0.5 * np.sin(i * 0.01) for _ in range(len(env.joint_indices))]
            env.set_joint_positions(target_positions)
            
            env.step_simulation()
            time.sleep(1/240)  # 240Hz仿真频率
            
    except KeyboardInterrupt:
        print("仿真中断")
    finally:
        env.close()