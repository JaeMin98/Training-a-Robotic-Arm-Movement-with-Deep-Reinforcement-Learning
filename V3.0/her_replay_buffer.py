import numpy as np
import random
from collections import namedtuple, deque
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HERReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed, k_future=4):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "goal"])
        self.seed = random.seed(seed)
        self.k_future = k_future
    
    def add(self, state, action, reward, next_state, done, goal):
        e = self.experience(state, action, reward, next_state, done, goal)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        goals = torch.from_numpy(np.vstack([e.goal for e in experiences if e is not None])).float().to(device)

        return (states, actions, rewards, next_states, dones, goals)

    def sample_her_batch(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        batch = []
        for experience in experiences:
            if experience is None:
                continue
            
            state, action, reward, next_state, done, goal = experience
            
            # HER: also use achieved goals as targets
            for _ in range(self.k_future):
                future = np.random.randint(len(self.memory))
                future_experience = self.memory[future]
                if future_experience is None:
                    continue
                future_state = future_experience.next_state
                
                achieved_goal = self.extract_goal(future_state)
                reward = self.compute_reward(next_state, achieved_goal)
                done = self.is_success(next_state, achieved_goal)
                
                batch.append((state, action, reward, next_state, done, achieved_goal))
        
        states, actions, rewards, next_states, dones, goals = zip(*batch)
        
        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).float().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)
        goals = torch.from_numpy(np.vstack(goals)).float().to(device)

        return (states, actions, rewards, next_states, dones, goals)

    def extract_goal(self, state):
        # Implement this based on your state representation
        # For example, if the goal is the end-effector position:
        return state[:3]  # Assuming the first 3 elements are x, y, z coordinates

    def compute_reward(self, state, goal):
        # Implement this based on your reward function
        # For example, negative distance between current state and goal
        return -np.linalg.norm(self.extract_goal(state) - goal)

    def is_success(self, state, goal):
        # Implement this based on your success criteria
        # For example, if the distance is less than a threshold
        return np.linalg.norm(self.extract_goal(state) - goal) < 0.05

    def __len__(self):
        return len(self.memory)