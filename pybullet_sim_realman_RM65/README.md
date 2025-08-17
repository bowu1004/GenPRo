# RM65机械臂仿真环境

基于PyBullet的RM65机械臂仿真环境，集成了强化学习（SAC算法）和传统运动学控制方法。(Additional doc: 仿真环境说明_Chinese.pdf in the same folder.)

## 项目结构

```
pybullet_sim/
├── rm65_env.py              # 基础仿真环境
├── rm65_controller.py       # 机械臂控制器
├── rm65_rl_env.py          # 强化学习环境
├── vision_encoder.py        # ResNet-18视觉编码器
├── sac_agent.py            # SAC算法实现
├── train_sac.py            # SAC训练脚本
├── evaluate_model.py       # 模型评估脚本
├── visualization_tools.py  # 可视化工具
├── requirements.txt        # 依赖包列表
└── README.md              # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 测试基础仿真环境

```python
from rm65_env import RM65Environment

# 创建仿真环境
env = RM65Environment()

# 测试环境
env.test_environment()
```

### 2. 强化学习训练

```bash
# 使用视觉输入训练SAC模型
python train_sac.py --use_vision --episodes 10000 --save_dir ./models

# 不使用视觉输入训练
python train_sac.py --episodes 5000 --save_dir ./models
```

### 3. 模型评估

```bash
# 评估训练好的模型
python evaluate_model.py --model_path ./models/sac_model_final.pth --episodes 100 --save_video
```


## 主要功能

### 1. 仿真环境 (rm65_env.py)
- PyBullet物理仿真
- RM65机械臂模型加载
- 关节状态控制和获取
- 相机图像获取
- 末端执行器位姿计算

### 2. 强化学习环境 (rm65_rl_env.py)
- 符合Gym接口的RL环境
- 可配置的观察空间（视觉/非视觉）
- 到达任务和抓取任务
- 自适应奖励函数

### 3. SAC算法 (sac_agent.py)
- Actor-Critic网络架构
- ResNet-18视觉编码集成
- 自动熵调节
- 经验回放缓冲区

### 4. 视觉编码 (vision_encoder.py)
- ResNet-18特征提取
- 视觉-状态融合
- 图像预处理管道

## 配置参数

### 训练配置
```python
config = TrainingConfig(
    episodes=10000,
    max_steps=200,
    batch_size=256,
    learning_rate=3e-4,
    use_vision=True,
    save_interval=1000
)
```

### 环境配置
```python
env_config = {
    'use_vision': True,
    'image_size': (224, 224),
    'max_episode_steps': 200,
    'reward_type': 'dense',
    'workspace_bounds': [[-0.5, 0.5], [-0.5, 0.5], [0.1, 0.8]]
}
```


### 完整训练流程
```python
from train_sac import SACTrainer, TrainingConfig

# 配置训练参数
config = TrainingConfig(
    episodes=5000,
    use_vision=True,
    save_dir='./models',
    log_dir='./logs'
)

# 创建训练器
trainer = SACTrainer(config)

# 开始训练
trainer.train()

# 评估模型
trainer.evaluate(num_episodes=100)
```
