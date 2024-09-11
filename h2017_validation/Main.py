import Env
import numpy as np
from ddpg_agent import Agent 
import torch


def ddpg():

    env = Env.Ned2_control()

    agent = Agent(state_size=6, action_size=3, random_seed=123456)

    agent.actor_local.load_state_dict(torch.load('trained_models/h2017_ddpg_actor_solved.pth'))
    agent.critic_local.load_state_dict(torch.load('trained_models/h2017_ddpg_critic_solved.pth'))

    
    total_success = []

    for current_level in range(env.MAX_Level_Of_Point+1):
        episode_success = []
        env.Level_Of_Point = current_level

        for episode in range(20):
            env.reset()
            states = env.get_state()
            agent.reset()

            scores = 0
            for timestep in range(200):
                actions = agent.act(np.array(states), add_noise=True)
                next_states, rewards, dones, success = env.step(actions)
                agent.step(np.array([states]), np.array([actions]), np.array([rewards]), np.array([next_states]), np.array([dones]), timestep)
                states = next_states
                scores += rewards
                if np.any(dones):
                    break

            episode_success.append(success)

        total_success.append((sum(episode_success)/len(episode_success)) * 100)
    print(total_success)



if __name__ == "__main__":
    ddpg()