import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import argparse
import json
from datetime import datetime
import cv2

from rm65_rl_env import RM65ReachEnv, RM65ManipulationEnv
from sac_agent import SACAgent
from vision_encoder import ImagePreprocessor
from train_sac import TrainingConfig

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, config: TrainingConfig, model_path: str, gui: bool = True):
        self.config = config
        self.gui = gui
        
        # 创建环境
        self.env = self._create_environment()
        
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
        
        # 加载模型
        self.agent.load(model_path)
        print(f"模型已加载: {model_path}")
        
        # 图像预处理器
        if config.use_vision:
            self.image_preprocessor = ImagePreprocessor(
                target_size=config.image_size
            )
            
        
        # 评估统计
        self.evaluation_results = []
        
    def _create_environment(self):
        """创建环境"""
        if self.config.env_name == "RM65Reach":
            return RM65ReachEnv(
                gui=self.gui,
                use_vision=self.config.use_vision,
                image_size=self.config.image_size,
                max_episode_steps=self.config.max_episode_steps
            )
        elif self.config.env_name == "RM65Manipulation":
            return RM65ManipulationEnv(
                gui=self.gui,
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
            # 添加批次维度 [channels, height, width] -> [1, channels, height, width]
            image = image.unsqueeze(0)
            processed_obs['image'] = image.to(self.config.device)
            
        return processed_obs
        
    def evaluate_episodes(self, num_episodes: int = 10, save_videos: bool = False) -> Dict:
        """评估多个回合"""
        results = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rates': [],
            'final_distances': [],
            'execution_times': [],
            'trajectories': []
        }
        
        video_frames = [] if save_videos else None
        
        for episode in range(num_episodes):
            print(f"\n评估回合 {episode + 1}/{num_episodes}")
            
            episode_result = self.evaluate_single_episode(
                episode_id=episode,
                record_trajectory=True,
                record_video=save_videos
            )
            
            # 收集结果
            results['episode_rewards'].append(episode_result['total_reward'])
            results['episode_lengths'].append(episode_result['episode_length'])
            results['success_rates'].append(float(episode_result['success']))
            results['final_distances'].append(episode_result['final_distance'])
            results['execution_times'].append(episode_result['execution_time'])
            results['trajectories'].append(episode_result['trajectory'])
            
            if save_videos and episode_result['video_frames']:
                video_frames.extend(episode_result['video_frames'])
                
            # 打印回合结果
            print(f"奖励: {episode_result['total_reward']:.2f}, "
                  f"步数: {episode_result['episode_length']}, "
                  f"成功: {episode_result['success']}, "
                  f"最终距离: {episode_result['final_distance']:.4f}m")
                  
        # 计算统计信息
        stats = self._calculate_statistics(results)
        
        # 保存视频
        if save_videos and video_frames:
            self._save_video(video_frames, "evaluation_video.mp4")
            
        return {'results': results, 'statistics': stats}
        
    def evaluate_single_episode(self, episode_id: int = 0, 
                              record_trajectory: bool = True,
                              record_video: bool = False) -> Dict:
        """评估单个回合"""
        obs = self.env.reset()
        total_reward = 0
        episode_length = 0
        trajectory = [] if record_trajectory else None
        video_frames = [] if record_video else None
        
        start_time = time.time()
        
        while episode_length < self.config.max_episode_steps:
            # 记录轨迹
            if record_trajectory:
                ee_pos, ee_ori = self.env.env.get_end_effector_pose()
                joint_states, _ = self.env.env.get_joint_states()
                trajectory.append({
                    'step': episode_length,
                    'joint_states': joint_states.copy(),
                    'ee_position': ee_pos.copy(),
                    'ee_orientation': ee_ori.copy(),
                    'target_position': self.env.target_position.copy()
                })
                
            # 记录视频帧
            if record_video:
                frame = self.env.env.get_camera_image()
                if frame is not None:
                    video_frames.append(frame)
                    
            # 选择动作
            processed_obs = self._preprocess_observation(obs)
            if self.config.use_vision and 'image' in processed_obs:
                action = self.agent.select_action(
                    processed_obs['state'].cpu().numpy(),
                    processed_obs['image'],
                    deterministic=True
                )
            else:
                action = self.agent.select_action(
                    processed_obs['state'].cpu().numpy(),
                    deterministic=True
                )
            
            # 执行动作
            next_obs, reward, done, info = self.env.step(action)
            total_reward += reward
            episode_length += 1
            
            obs = next_obs
            
            if done:
                break
                
        execution_time = time.time() - start_time
        
        # 计算最终距离
        ee_pos, _ = self.env.env.get_end_effector_pose()
        final_distance = np.linalg.norm(ee_pos - self.env.target_position)
        
        return {
            'episode_id': episode_id,
            'total_reward': total_reward,
            'episode_length': episode_length,
            'success': info.get('success', False),
            'final_distance': final_distance,
            'execution_time': execution_time,
            'trajectory': trajectory,
            'video_frames': video_frames
        }
        
    
    def _calculate_statistics(self, results: Dict) -> Dict:
        """计算统计信息"""
        stats = {}
        
        for key, values in results.items():
            if key == 'trajectories':
                continue
                
            if values:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)
                stats[f'{key}_min'] = np.min(values)
                stats[f'{key}_max'] = np.max(values)
                
        return stats
        
    def _save_video(self, frames: List[np.ndarray], filename: str):
        """保存视频"""
        if not frames:
            return
            
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
        
        for frame in frames:
            # 转换颜色格式
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
            
        out.release()
        print(f"视频已保存: {filename}")
        
    def plot_trajectory(self, trajectory: List[Dict], save_path: str = None):
        """绘制轨迹图"""
        if not trajectory:
            return
            
        # 提取数据
        steps = [t['step'] for t in trajectory]
        ee_positions = np.array([t['ee_position'] for t in trajectory])
        target_pos = trajectory[0]['target_position']
        
        # 创建图表
        fig = plt.figure(figsize=(15, 10))
        
        # 3D轨迹图
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 
                'b-', linewidth=2, label='末端执行器轨迹')
        ax1.scatter(*target_pos, color='red', s=100, label='目标位置')
        ax1.scatter(*ee_positions[0], color='green', s=100, label='起始位置')
        ax1.scatter(*ee_positions[-1], color='orange', s=100, label='结束位置')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D轨迹')
        ax1.legend()
        
        # 距离随时间变化
        ax2 = fig.add_subplot(222)
        distances = [np.linalg.norm(t['ee_position'] - target_pos) for t in trajectory]
        ax2.plot(steps, distances, 'r-', linewidth=2)
        ax2.set_xlabel('步数')
        ax2.set_ylabel('距离目标 (m)')
        ax2.set_title('距离目标随时间变化')
        ax2.grid(True)
        
        # XY平面投影
        ax3 = fig.add_subplot(223)
        ax3.plot(ee_positions[:, 0], ee_positions[:, 1], 'b-', linewidth=2, label='轨迹')
        ax3.scatter(target_pos[0], target_pos[1], color='red', s=100, label='目标')
        ax3.scatter(ee_positions[0, 0], ee_positions[0, 1], color='green', s=100, label='起始')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_title('XY平面投影')
        ax3.legend()
        ax3.grid(True)
        
        # 关节角度变化
        ax4 = fig.add_subplot(224)
        joint_angles = np.array([t['joint_states'] for t in trajectory])
        for i in range(min(6, joint_angles.shape[1])):
            ax4.plot(steps, joint_angles[:, i], label=f'关节{i+1}')
        ax4.set_xlabel('步数')
        ax4.set_ylabel('关节角度 (rad)')
        ax4.set_title('关节角度变化')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"轨迹图已保存: {save_path}")
        else:
            plt.show()
            
        plt.close()
        
    def generate_report(self, evaluation_data: Dict, output_dir: str = "./evaluation_results"):
        """生成评估报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细结果
        results_file = os.path.join(output_dir, "evaluation_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            # 转换numpy数组为列表以便JSON序列化
            serializable_data = self._make_serializable(evaluation_data)
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
        # 生成统计图表
        self._plot_evaluation_statistics(evaluation_data, output_dir)
        
        self._generate_text_report(evaluation_data, output_dir)
        
        print(f"评估报告已生成: {output_dir}")
        
    def _make_serializable(self, obj):
        """使对象可JSON序列化"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
            
    def _plot_evaluation_statistics(self, data: Dict, output_dir: str):
        """绘制评估统计图表"""
        results = data['results']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 奖励分布
        axes[0, 0].hist(results['episode_rewards'], bins=20, alpha=0.7, color='blue')
        axes[0, 0].set_title('回合奖励分布')
        axes[0, 0].set_xlabel('奖励')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].grid(True)
        
        # 回合长度分布
        axes[0, 1].hist(results['episode_lengths'], bins=20, alpha=0.7, color='green')
        axes[0, 1].set_title('回合长度分布')
        axes[0, 1].set_xlabel('步数')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].grid(True)
        
        # 最终距离分布
        axes[1, 0].hist(results['final_distances'], bins=20, alpha=0.7, color='red')
        axes[1, 0].set_title('最终距离分布')
        axes[1, 0].set_xlabel('距离 (m)')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].grid(True)
        
        # 成功率
        success_rate = np.mean(results['success_rates'])
        axes[1, 1].bar(['成功', '失败'], [success_rate, 1-success_rate], 
                      color=['green', 'red'], alpha=0.7)
        axes[1, 1].set_title(f'成功率: {success_rate:.1%}')
        axes[1, 1].set_ylabel('比例')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "evaluation_statistics.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def _generate_text_report(self, data: Dict, output_dir: str):
        """生成文本报告"""
        stats = data['statistics']
        
        report = f"""
# RM65机械臂SAC模型评估报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 评估配置
- 环境: {self.config.env_name}
- 使用视觉: {self.config.use_vision}
- 最大回合步数: {self.config.max_episode_steps}
- 评估回合数: {len(data['results']['episode_rewards'])}

## 性能统计

### 回合奖励
- 平均值: {stats.get('episode_rewards_mean', 0):.2f}
- 标准差: {stats.get('episode_rewards_std', 0):.2f}
- 最小值: {stats.get('episode_rewards_min', 0):.2f}
- 最大值: {stats.get('episode_rewards_max', 0):.2f}

### 回合长度
- 平均值: {stats.get('episode_lengths_mean', 0):.1f} 步
- 标准差: {stats.get('episode_lengths_std', 0):.1f}
- 最小值: {stats.get('episode_lengths_min', 0)} 步
- 最大值: {stats.get('episode_lengths_max', 0)} 步

### 成功率
- 成功率: {stats.get('success_rates_mean', 0):.1%}

### 最终距离
- 平均值: {stats.get('final_distances_mean', 0):.4f} m
- 标准差: {stats.get('final_distances_std', 0):.4f}
- 最小值: {stats.get('final_distances_min', 0):.4f} m
- 最大值: {stats.get('final_distances_max', 0):.4f} m

### 执行时间
- 平均值: {stats.get('execution_times_mean', 0):.3f} s
- 标准差: {stats.get('execution_times_std', 0):.3f}
"""
        
        # 添加对比结果（如果有）
        if 'comparison_summary' in data:
            comp = data['comparison_summary']
            report += f"""

## 与传统运动学方法对比

### 成功率对比
- 强化学习: {comp['rl_success_rate']:.1%}
- 逆运动学: {comp['ik_success_rate']:.1%}

### 平均距离对比
- 强化学习: {comp['rl_avg_distance']:.4f} m
- 逆运动学: {comp['ik_avg_distance']:.4f} m

### 平均执行时间对比
- 强化学习: {comp['rl_avg_time']:.3f} s
- 逆运动学: {comp['ik_avg_time']:.3f} s

### 逆运动学可解率
"""
        
        report_file = os.path.join(output_dir, "evaluation_report.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
    def close(self):
        """清理资源"""
        self.env.close()

def main():
    parser = argparse.ArgumentParser(description='RM65机械臂模型评估')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--episodes', type=int, default=10, help='评估回合数')
    parser.add_argument('--gui', action='store_true', help='显示GUI')
    parser.add_argument('--save-video', action='store_true', help='保存视频')
    parser.add_argument('--compare-ik', action='store_true', help='与逆运动学对比')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results', help='输出目录')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = TrainingConfig.load_config(args.config)
    else:
        # 尝试从模型目录加载配置
        model_dir = os.path.dirname(args.model)
        config_path = os.path.join(model_dir, '..', 'config.json')
        if os.path.exists(config_path):
            config = TrainingConfig.load_config(config_path)
        else:
            config = TrainingConfig()
            print("警告: 未找到配置文件，使用默认配置")
            
    # 创建评估器
    evaluator = ModelEvaluator(config, args.model, gui=args.gui)
    
    try:
        print(f"开始评估模型: {args.model}")
        
        # 评估多个回合
        evaluation_data = evaluator.evaluate_episodes(
            num_episodes=args.episodes,
            save_videos=args.save_video
        )
        
            
        # 生成报告
        evaluator.generate_report(evaluation_data, args.output_dir)
        
        # 绘制轨迹图（第一个回合）
        if evaluation_data['results']['trajectories']:
            trajectory_path = os.path.join(args.output_dir, "trajectory_plot.png")
            evaluator.plot_trajectory(
                evaluation_data['results']['trajectories'][0],
                trajectory_path
            )
            
        print("\n评估完成！")
        print(f"结果保存在: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\n评估被用户中断")
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        evaluator.close()
        
if __name__ == "__main__":
    main()