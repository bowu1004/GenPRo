import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Tuple, Dict, Any, Optional
from vision_encoder import ResNet18Encoder, VisionStateEncoder

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, 
                 use_vision: bool = False, image_shape: Tuple[int, int, int] = (224, 224, 3)):
        """
        初始化回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            state_dim: 状态维度
            action_dim: 动作维度
            use_vision: 是否使用视觉
            image_shape: 图像形状
        """
        self.capacity = capacity
        self.use_vision = use_vision
        self.position = 0
        self.size = 0
        
        # 存储数组
        if use_vision:
            self.images = np.zeros((capacity, *image_shape), dtype=np.uint8)
            self.next_images = np.zeros((capacity, *image_shape), dtype=np.uint8)
            
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.bool_)
        
    def push(self, state, action, reward, next_state, done, 
             image=None, next_image=None):
        """添加经验"""
        if self.use_vision:
            self.images[self.position] = image
            self.next_images[self.position] = next_image
            
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """采样批量经验"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        batch = {
            'states': torch.FloatTensor(self.states[indices]),
            'actions': torch.FloatTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'next_states': torch.FloatTensor(self.next_states[indices]),
            'dones': torch.BoolTensor(self.dones[indices])
        }
        
        if self.use_vision:
            batch['images'] = torch.FloatTensor(self.images[indices]) / 255.0
            batch['next_images'] = torch.FloatTensor(self.next_images[indices]) / 255.0
            
        return batch
        
    def __len__(self):
        return self.size


class Actor(nn.Module):
    """SAC Actor网络"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 use_vision: bool = False,
                 vision_feature_dim: int = 512,
                 log_std_min: float = -20,
                 log_std_max: float = 2):
        """
        初始化Actor网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            use_vision: 是否使用视觉
            vision_feature_dim: 视觉特征维度
            log_std_min: 对数标准差最小值
            log_std_max: 对数标准差最大值
        """
        super(Actor, self).__init__()
        
        self.use_vision = use_vision
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        if use_vision:
            # 视觉编码器
            self.vision_encoder = ResNet18Encoder(feature_dim=vision_feature_dim)
            self.vision_state_encoder = VisionStateEncoder(
                vision_feature_dim=vision_feature_dim,
                state_dim=state_dim,
                output_dim=hidden_dim
            )
            input_dim = hidden_dim
        else:
            input_dim = state_dim
            
        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 均值和标准差输出层
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state, image=None):
        """
        前向传播
        
        Args:
            state: 状态向量
            image: 图像（如果使用视觉）
            
        Returns:
            mean: 动作均值
            log_std: 动作对数标准差
        """
        if self.use_vision and image is not None:
            # 提取视觉特征
            vision_features = self.vision_encoder.encode_image(image)
            # 融合视觉和状态特征
            x = self.vision_state_encoder(vision_features, state)
        else:
            x = state
            
        # 通过策略网络
        x = self.policy_net(x)
        
        # 计算均值和对数标准差
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
        
    def sample(self, state, image=None):
        """
        采样动作
        
        Args:
            state: 状态向量
            image: 图像（如果使用视觉）
            
        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率
        """
        mean, log_std = self.forward(state, image)
        std = log_std.exp()
        
        # 重参数化技巧
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # 重参数化采样
        
        # 应用tanh激活函数
        action = torch.tanh(x_t)
        
        # 计算对数概率
        log_prob = normal.log_prob(x_t)
        # 修正tanh变换的雅可比行列式
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob
        
    def get_action(self, state, image=None, deterministic=False):
        """
        获取动作（用于推理）
        
        Args:
            state: 状态向量
            image: 图像（如果使用视觉）
            deterministic: 是否确定性输出
            
        Returns:
            action: 动作
        """
        with torch.no_grad():
            if deterministic:
                mean, _ = self.forward(state, image)
                action = torch.tanh(mean)
            else:
                action, _ = self.sample(state, image)
                
        return action


class Critic(nn.Module):
    """SAC Critic网络（Q网络）"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 use_vision: bool = False,
                 vision_feature_dim: int = 512):
        """
        初始化Critic网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            use_vision: 是否使用视觉
            vision_feature_dim: 视觉特征维度
        """
        super(Critic, self).__init__()
        
        self.use_vision = use_vision
        
        if use_vision:
            # 视觉编码器
            self.vision_encoder = ResNet18Encoder(feature_dim=vision_feature_dim)
            self.vision_state_encoder = VisionStateEncoder(
                vision_feature_dim=vision_feature_dim,
                state_dim=state_dim,
                output_dim=hidden_dim
            )
            input_dim = hidden_dim + action_dim
        else:
            input_dim = state_dim + action_dim
            
        # Q网络
        self.q_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action, image=None):
        """
        前向传播
        
        Args:
            state: 状态向量
            action: 动作向量
            image: 图像（如果使用视觉）
            
        Returns:
            q_value: Q值
        """
        if self.use_vision and image is not None:
            # 提取视觉特征
            vision_features = self.vision_encoder.encode_image(image)
            # 融合视觉和状态特征
            state_features = self.vision_state_encoder(vision_features, state)
            x = torch.cat([state_features, action], dim=1)
        else:
            x = torch.cat([state, action], dim=1)
            
        q_value = self.q_net(x)
        return q_value


class SACAgent:
    """SAC智能体"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 auto_entropy_tuning: bool = True,
                 use_vision: bool = False,
                 vision_feature_dim: int = 512,
                 device: str = 'cpu'):
        """
        初始化SAC智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            lr: 学习率
            gamma: 折扣因子
            tau: 软更新系数
            alpha: 熵正则化系数
            auto_entropy_tuning: 是否自动调节熵系数
            use_vision: 是否使用视觉
            vision_feature_dim: 视觉特征维度
            device: 设备
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy_tuning = auto_entropy_tuning
        self.use_vision = use_vision
        
        # 创建网络
        self.actor = Actor(
            state_dim, action_dim, hidden_dim, 
            use_vision, vision_feature_dim
        ).to(device)
        
        self.critic1 = Critic(
            state_dim, action_dim, hidden_dim,
            use_vision, vision_feature_dim
        ).to(device)
        
        self.critic2 = Critic(
            state_dim, action_dim, hidden_dim,
            use_vision, vision_feature_dim
        ).to(device)
        
        # 目标网络
        self.target_critic1 = Critic(
            state_dim, action_dim, hidden_dim,
            use_vision, vision_feature_dim
        ).to(device)
        
        self.target_critic2 = Critic(
            state_dim, action_dim, hidden_dim,
            use_vision, vision_feature_dim
        ).to(device)
        
        # 复制参数到目标网络
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # 自动熵调节
        if auto_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            
        # 初始化回放缓冲区
        self.replay_buffer = ReplayBuffer(
            capacity=100000,  # 默认容量
            state_dim=state_dim,
            action_dim=action_dim,
            use_vision=use_vision
        )
            
    def select_action(self, state, image=None, deterministic=False):
        """
        选择动作
        
        Args:
            state: 状态
            image: 图像（如果使用视觉）
            deterministic: 是否确定性
            
        Returns:
            action: 动作
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if self.use_vision and image is not None:
            if isinstance(image, np.ndarray):
                image = torch.FloatTensor(image).unsqueeze(0).to(self.device)
                if image.max() > 1.0:
                    image = image / 255.0
            else:
                image = image.to(self.device)
                
        action = self.actor.get_action(state, image, deterministic)
        return action.cpu().numpy()[0]
        
    def update(self, replay_buffer, batch_size=256):
        """
        更新网络参数
        
        Args:
            replay_buffer: 回放缓冲区
            batch_size: 批量大小
            
        Returns:
            losses: 损失字典
        """
        # 采样批量数据
        batch = replay_buffer.sample(batch_size)
        
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        images = None
        next_images = None
        if self.use_vision:
            images = batch['images'].to(self.device)
            next_images = batch['next_images'].to(self.device)
            
        # 更新Critic网络
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states, next_images)
            target_q1 = self.target_critic1(next_states, next_actions, next_images)
            target_q2 = self.target_critic2(next_states, next_actions, next_images)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones.float()) * self.gamma * target_q
            
        current_q1 = self.critic1(states, actions, images)
        current_q2 = self.critic2(states, actions, images)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # 更新Actor网络
        new_actions, log_probs = self.actor.sample(states, images)
        q1_new = self.critic1(states, new_actions, images)
        q2_new = self.critic2(states, new_actions, images)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新熵系数
        alpha_loss = 0
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
            
        # 软更新目标网络
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if self.auto_entropy_tuning else 0,
            'alpha': self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha
        }
        
    def update_parameters(self, batch_size=256):
        """更新网络参数（兼容接口）"""
        losses = self.update(self.replay_buffer, batch_size)
        return losses['critic1_loss'], losses['actor_loss'], losses.get('alpha_loss', 0)
        
    def _soft_update(self, target, source):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
            
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.auto_entropy_tuning else None
        }, filepath)
        
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        
        if self.auto_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])


if __name__ == "__main__":
    # 测试SAC智能体
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建智能体
    agent = SACAgent(
        state_dim=20,
        action_dim=6,
        use_vision=True,
        device=device
    )
    
    # 创建回放缓冲区
    replay_buffer = ReplayBuffer(
        capacity=100000,
        state_dim=20,
        action_dim=6,
        use_vision=True
    )
    
    # 测试动作选择
    test_state = np.random.randn(20)
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    action = agent.select_action(test_state, test_image)
    print(f"选择的动作: {action}")
    
    # 添加一些经验到回放缓冲区
    for _ in range(1000):
        state = np.random.randn(20)
        action = np.random.randn(6)
        reward = np.random.randn()
        next_state = np.random.randn(20)
        done = np.random.choice([True, False])
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        next_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        replay_buffer.push(state, action, reward, next_state, done, image, next_image)
        
    # 测试更新
    if len(replay_buffer) >= 256:
        losses = agent.update(replay_buffer, batch_size=256)
        print(f"损失: {losses}")
        
    print("SAC智能体测试完成！")