import Env
import numpy as np
from ddpg_agent import Agent 
from collections import deque
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import wandb

wandb.init(project='Refer2')
wandb.run.name = 'DDPG_V2.0'
wandb.run.save()

env = Env.Ned2_control()
agent = Agent(state_size=6, action_size=3, random_seed=123456)

episode_success, success_rate_list = [], []

def ddpg(n_episodes=100000, max_t=200):
    for i_episode in range(1, n_episodes+1):
        env.reset()
        states = env.get_state()
        agent.reset()

        scores = 0
        for t in range(max_t):
            actions = agent.act(np.array(states))
            next_states, rewards, dones, success = env.step(actions)  # 새로운 env.step() 반환 값에 맞춰 수정
            agent.step(np.array([states]), np.array([actions]), np.array([rewards]), np.array([next_states]), np.array([dones]))
            states = next_states
            scores += rewards
            if np.any(dones):
                break

        episode_success.append(success)
        success_rate = np.mean(episode_success[-min(10, len(episode_success)):])
        success_rate_list.append(success_rate)
        wandb.log({'success_rate':success_rate}, step=i_episode)

        print(f"Episode: {i_episode}, Reward: {scores}")
        wandb.log({'episode_reward':scores}, step=i_episode)
        wandb.log({'memory size':len(agent.memory)}, step=i_episode)

        if(len(success_rate_list) > 4):
            if np.mean(success_rate_list[-min(5, len(success_rate_list)):]) >= 0.9:
                torch.save(agent.actor_local.state_dict(), 'actor_solved.pth')
                torch.save(agent.critic_local.state_dict(), 'critic_solved.pth')
                break



if __name__ == "__main__":
    ddpg()