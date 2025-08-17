import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import argparse
import json
from datetime import datetime

from rm65_rl_env import RM65ReachEnv, RM65ManipulationEnv
from sac_agent import SACAgent
from vision_encoder import ImagePreprocessor

class TrainingConfig:
    """训练配置类"""
    
    def __init__(self):
        # 环境配置
        self.env_name = "RM65Reach"
        self.use_vision = True
        self.image_size = (224, 224)
        self.max_episode_steps = 200
        self.target_update_freq = 1000
        
        # 训练配置
        self.total_timesteps = 1000000
        self.learning_starts = 10000
        self.batch_size = 256
        self.buffer_size = 1000000
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.auto_entropy_tuning = True
        
        # 网络配置
        self.hidden_dim = 256
        self.vision_feature_dim = 512
        self.state_dim = 24  # 关节角度
        self.action_dim = 6  # 关节角度增量
        
        # 评估配置
        self.eval_freq = 10000
        self.eval_episodes = 10
        
        # 保存配置
        self.save_freq = 50000
        self.log_freq = 1000
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 输出目录
        self.output_dir = "./results"
        self.model_dir = "./models"
        self.log_dir = "./logs"
        
    def save_config(self, filepath: str):
        """保存配置到文件"""
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_') and not callable(v)}
        config_dict['device'] = str(config_dict['device'])
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
    @classmethod
    def load_config(cls, filepath: str):
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            
        config = cls()
        for key, value in config_dict.items():
            if key == 'device':
                setattr(config, key, torch.device(value))
            else:
                setattr(config, key, value)
        return config

class SACTrainer:
    """SAC训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # 创建目录
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.model_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # 创建环境
        self.env = self._create_environment()
        self.eval_env = self._create_environment()
        
        # 创建智能体
        self.agent = SACAgent(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            lr=config.learning_rate,
            gamma=config.gamma,
            tau=config.tau,
            alpha=config.alpha,
            auto_entropy_tuning=config.auto_entropy_tuning,
            device=config.device,
            use_vision=config.use_vision,
            vision_feature_dim=config.vision_feature_dim
        )
        
        # 图像预处理器
        if config.use_vision:
            self.image_preprocessor = ImagePreprocessor(
                target_size=config.image_size
            )
        
        # 日志记录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f"{config.log_dir}/sac_{timestamp}")
        
        # 训练统计
        self.total_steps = 0
        self.episode_count = 0
        self.best_eval_reward = -np.inf
        
        # 保存配置
        config.save_config(os.path.join(config.output_dir, "config.json"))
        
    def _create_environment(self):
        """创建环境"""
        if self.config.env_name == "RM65Reach":
            return RM65ReachEnv(
                gui=False,
                use_vision=self.config.use_vision,
                image_size=self.config.image_size,
                max_episode_steps=self.config.max_episode_steps
            )
        elif self.config.env_name == "RM65Manipulation":
            return RM65ManipulationEnv(
                gui=False,
                use_vision=self.config.use_vision,
                image_size=self.config.image_size,
                max_episode_steps=self.config.max_episode_steps
            )
        else:
            raise ValueError(f"未知环境: {self.config.env_name}")
            
    def _preprocess_observation(self, obs: Dict) -> Dict:
        """预处理观察"""
        processed_obs = {}
        
        # 处理状态
        processed_obs['state'] = torch.FloatTensor(obs['state']).to(self.config.device)
        
        # 处理图像（如果使用视觉）
        if self.config.use_vision and 'image' in obs:
            image = self.image_preprocessor.preprocess(obs['image'])
            processed_obs['image'] = image.to(self.config.device)
            
        return processed_obs
        
    def train(self):
        """主训练循环"""
        print(f"开始训练，目标步数: {self.config.total_timesteps}")
        print(f"设备: {self.config.device}")
        print(f"使用视觉: {self.config.use_vision}")
        
        # 初始化环境
        obs = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_start_time = time.time()
        
        # 训练循环
        for step in range(self.config.total_timesteps):
            self.total_steps = step
            
            # 选择动作
            if step < self.config.learning_starts:
                # 随机探索
                action = self.env.action_space.sample()
            else:
                # 使用策略选择动作
                if self.config.use_vision:
                    state = obs['state']
                    image = obs.get('image', None)
                    action = self.agent.select_action(state, image, deterministic=False)
                else:
                    action = self.agent.select_action(obs, deterministic=False)
                
            # 执行动作
            next_obs, reward, done, info = self.env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            # 存储经验
            if self.config.use_vision:
                # 提取图像数据
                current_image = obs.get('image', None)
                next_image = next_obs.get('image', None)
                
                self.agent.replay_buffer.push(
                    obs['state'], action, reward, next_obs['state'], done,
                    current_image, next_image
                )
            else:
                self.agent.replay_buffer.push(
                    obs, action, reward, next_obs, done
                )
            
            # 更新观察
            obs = next_obs
            
            # 训练智能体
            if step >= self.config.learning_starts:
                critic_loss, actor_loss, alpha_loss = self.agent.update_parameters(
                    self.config.batch_size
                )
                
                # 记录损失
                if step % self.config.log_freq == 0:
                    self.writer.add_scalar('Loss/Critic', critic_loss, step)
                    self.writer.add_scalar('Loss/Actor', actor_loss, step)
                    if alpha_loss is not None:
                        self.writer.add_scalar('Loss/Alpha', alpha_loss, step)
                        self.writer.add_scalar('Alpha', self.agent.alpha, step)
                        
            # 处理回合结束
            if done or episode_steps >= self.config.max_episode_steps:
                # 记录回合统计
                episode_time = time.time() - episode_start_time
                self.episode_count += 1
                
                self.writer.add_scalar('Episode/Reward', episode_reward, self.episode_count)
                self.writer.add_scalar('Episode/Steps', episode_steps, self.episode_count)
                self.writer.add_scalar('Episode/Time', episode_time, self.episode_count)
                
                if 'success' in info:
                    self.writer.add_scalar('Episode/Success', float(info['success']), self.episode_count)
                    
                # 打印进度
                if self.episode_count % 10 == 0:
                    print(f"Episode {self.episode_count}, Step {step}, "
                          f"Reward: {episode_reward:.2f}, Steps: {episode_steps}, "
                          f"Time: {episode_time:.2f}s")
                    
                # 重置环境
                obs = self.env.reset()
                episode_reward = 0
                episode_steps = 0
                episode_start_time = time.time()
                
            # 评估
            if step % self.config.eval_freq == 0 and step > 0:
                eval_reward = self.evaluate()
                self.writer.add_scalar('Eval/Reward', eval_reward, step)
                
                # 保存最佳模型
                if eval_reward > self.best_eval_reward:
                    self.best_eval_reward = eval_reward
                    self.save_model("best_model.pth")
                    print(f"新的最佳模型保存，评估奖励: {eval_reward:.2f}")
                    
            # 定期保存模型
            if step % self.config.save_freq == 0 and step > 0:
                self.save_model(f"model_step_{step}.pth")
                
        # 训练结束，保存最终模型
        self.save_model("final_model.pth")
        print("训练完成！")
        
    def evaluate(self, num_episodes: Optional[int] = None) -> float:
        """评估智能体性能"""
        if num_episodes is None:
            num_episodes = self.config.eval_episodes
            
        total_reward = 0
        success_count = 0
        
        for episode in range(num_episodes):
            obs = self.eval_env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < self.config.max_episode_steps:
                if self.config.use_vision:
                    state = obs['state']
                    image = obs.get('image', None)
                    action = self.agent.select_action(state, image, deterministic=True)
                else:
                    action = self.agent.select_action(obs, deterministic=True)
                
                obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward
                steps += 1
                
            total_reward += episode_reward
            if 'success' in info and info['success']:
                success_count += 1
                
        avg_reward = total_reward / num_episodes
        success_rate = success_count / num_episodes
        
        print(f"评估结果: 平均奖励 {avg_reward:.2f}, 成功率 {success_rate:.2%}")
        
        self.writer.add_scalar('Eval/Reward', avg_reward, self.total_steps)
        self.writer.add_scalar('Eval/SuccessRate', success_rate, self.total_steps)
        
        return avg_reward
        
    def save_model(self, filename: str):
        """保存模型"""
        filepath = os.path.join(self.config.model_dir, filename)
        self.agent.save(filepath)
        
    def load_model(self, filename: str):
        """加载模型"""
        filepath = os.path.join(self.config.model_dir, filename)
        self.agent.load(filepath)
        
    def close(self):
        """清理资源"""
        self.env.close()
        self.eval_env.close()
        self.writer.close()

def create_learning_curve_plot(log_dir: str, output_path: str):
    """创建学习曲线图"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        # 读取tensorboard日志
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        
        # 获取数据
        episode_rewards = event_acc.Scalars('Episode/Reward') if 'Episode/Reward' in event_acc.Tags()['scalars'] else []
        eval_rewards = event_acc.Scalars('Eval/Reward') if 'Eval/Reward' in event_acc.Tags()['scalars'] else []
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 训练奖励
        if episode_rewards:
            steps = [x.step for x in episode_rewards]
            rewards = [x.value for x in episode_rewards]
            ax1.plot(steps, rewards, alpha=0.3, color='blue')
            
            # 平滑曲线
            window_size = min(100, len(rewards) // 10)
            if window_size > 1:
                smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                ax1.plot(steps[window_size-1:], smoothed, color='blue', linewidth=2)
                
        ax1.set_title('训练奖励')
        ax1.set_xlabel('回合')
        ax1.set_ylabel('奖励')
        ax1.grid(True)
        
        # 评估奖励
        if eval_rewards:
            eval_steps = [x.step for x in eval_rewards]
            eval_values = [x.value for x in eval_rewards]
            ax2.plot(eval_steps, eval_values, 'ro-', linewidth=2, markersize=4)
            
        ax2.set_title('评估奖励')
        ax2.set_xlabel('训练步数')
        ax2.set_ylabel('平均奖励')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"学习曲线图已保存到: {output_path}")
        
    except ImportError:
        print("需要安装tensorboard来生成学习曲线图")
    except Exception as e:
        print(f"生成学习曲线图时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='RM65机械臂SAC训练')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--env', type=str, default='RM65Reach', 
                       choices=['RM65Reach', 'RM65Manipulation'],
                       help='环境类型')
    parser.add_argument('--vision', action='store_true', help='使用视觉输入')
    parser.add_argument('--no-vision', action='store_true', help='不使用视觉输入')
    parser.add_argument('--timesteps', type=int, default=1000000, help='训练步数')
    parser.add_argument('--eval-only', action='store_true', help='仅评估模式')
    parser.add_argument('--model', type=str, help='加载的模型文件')
    parser.add_argument('--output-dir', type=str, default='./results', help='输出目录')
    
    args = parser.parse_args()
    
    # 加载或创建配置
    if args.config:
        config = TrainingConfig.load_config(args.config)
    else:
        config = TrainingConfig()
        
    # 应用命令行参数
    if args.env:
        config.env_name = args.env
    if args.vision:
        config.use_vision = True
    if args.no_vision:
        config.use_vision = False
    if args.timesteps:
        config.total_timesteps = args.timesteps
    if args.output_dir:
        config.output_dir = args.output_dir
        config.model_dir = os.path.join(args.output_dir, "models")
        config.log_dir = os.path.join(args.output_dir, "logs")
        
    # 创建训练器
    trainer = SACTrainer(config)
    
    try:
        if args.eval_only:
            # 仅评估模式
            if args.model:
                trainer.load_model(args.model)
                print(f"加载模型: {args.model}")
            else:
                print("评估模式需要指定模型文件")
                return
                
            # 运行评估
            avg_reward = trainer.evaluate(num_episodes=20)
            print(f"评估完成，平均奖励: {avg_reward:.2f}")
            
        else:
            # 训练模式
            if args.model:
                trainer.load_model(args.model)
                print(f"从模型继续训练: {args.model}")
                
            # 开始训练
            trainer.train()
            
            # 生成学习曲线
            log_dirs = [d for d in os.listdir(config.log_dir) 
                       if d.startswith('sac_') and os.path.isdir(os.path.join(config.log_dir, d))]
            if log_dirs:
                latest_log_dir = os.path.join(config.log_dir, sorted(log_dirs)[-1])
                curve_path = os.path.join(config.output_dir, "learning_curve.png")
                create_learning_curve_plot(latest_log_dir, curve_path)
                
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        trainer.close()
        
if __name__ == "__main__":
    main()