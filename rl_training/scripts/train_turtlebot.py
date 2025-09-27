import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gym_gazebo2
import gym_gazebo_maze
from something import GazeboMazeTurtleBotEnv  

env = GazeboMazeTurtleBotEnv()

print("Script started .....")

# Define Q-Network (Neural Network for DQN)
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# Training Parameters
EPISODES = 1000
GAMMA = 0.99
LR = 0.0005  # Reduced learning rate for stability
BATCH_SIZE = 32  # Smaller batch size
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
MEMORY_SIZE = 50000  # Increased memory size
UPDATE_TARGET_EVERY = 100  # Less frequent target updates
WARMUP_STEPS = 1000  # Steps before training starts

# Get environment dimensions
state_size = np.prod(env.observation_space.shape)
action_size = env.action_space.n

print(f"State size: {state_size}, Action size: {action_size}")

# Initialize networks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(state_size, action_size).to(device)
target_model = DQN(state_size, action_size).to(device)
target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=LR)
memory = deque(maxlen=MEMORY_SIZE)

# Tracking variables
episode_rewards = []
step_count = 0

def preprocess_state(state):
    """Preprocess and normalize the state"""
    if isinstance(state, dict):
        # If state is a dictionary, extract relevant features
        lidar_data = state.get('scan', [])
        position = state.get('position', [0, 0])
        orientation = state.get('orientation', [0])
        state = np.concatenate([lidar_data, position, orientation])
    
    state = np.array(state, dtype=np.float32)
    state = np.clip(state, -10, 10)  # Clip extreme values
    state = state / 10.0  # Normalize
    return state

def train_model():
    """Training function with improved stability"""
    if len(memory) < WARMUP_STEPS:
        return
    
    batch = random.sample(memory, BATCH_SIZE)
    states = torch.FloatTensor([e[0] for e in batch]).to(device)
    actions = torch.LongTensor([e[1] for e in batch]).to(device)
    rewards = torch.FloatTensor([e[2] for e in batch]).to(device)
    next_states = torch.FloatTensor([e[3] for e in batch]).to(device)
    dones = torch.BoolTensor([e[4] for e in batch]).to(device)
    
    current_q_values = model(states).gather(1, actions.unsqueeze(1))
    next_q_values = target_model(next_states).max(1)[0].detach()
    target_q_values = rewards + (GAMMA * next_q_values * (~dones))
    
    loss = nn.functional.huber_loss(current_q_values.squeeze(), target_q_values)
    
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()

# Training loop
for episode in range(EPISODES):
    state, info = env.reset()
    state = preprocess_state(state)
    state = np.reshape(state, [1, -1])
    
    total_reward = 0
    steps_in_episode = 0
    max_steps = 1000  # Prevent infinite episodes

    while steps_in_episode < max_steps:
        # Epsilon-greedy action selection
        if np.random.rand() < EPSILON:
            action = np.random.choice(action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

        # Take action in environment
        try:
            step_result = env.step(action)
            
            # Handle different return formats
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
            elif len(step_result) == 6:
                next_state, reward, terminated, truncated, info, _ = step_result
            else:
                print(f"Unexpected step return format: {len(step_result)} elements")
                break
            
            next_state = preprocess_state(next_state)
            next_state = np.reshape(next_state, [1, -1])
            done = terminated or truncated
            
            # Enhanced reward shaping
            if isinstance(info, dict):
                if 'goal_distance' in info:
                    goal_distance = info['goal_distance']
                    if goal_distance < 0.3:
                        reward += 100  # Large reward for reaching goal
                        print(f"[INFO] Goal reached! Distance: {goal_distance:.3f}")
                    elif goal_distance < 1.0:
                        reward += 10  # Moderate reward for getting close
                    
                    # Distance-based reward shaping
                    reward += max(0, (2.0 - goal_distance) * 0.1)
                
                if 'collision' in info and info['collision']:
                    reward -= 50  # Penalty for collision
                    print("[INFO] Collision detected!")
            
            # Small penalty for each step to encourage efficiency
            reward -= 0.01
            
            # Store experience
            memory.append((state.flatten(), action, reward, next_state.flatten(), done))
            
            # Train the model
            if step_count > WARMUP_STEPS and step_count % 4 == 0:
                loss = train_model()
                if loss is not None and step_count % 1000 == 0:
                    print(f"Step {step_count}, Loss: {loss:.4f}")
            
            state = next_state
            total_reward += reward
            steps_in_episode += 1
            step_count += 1
            
            if done:
                break
                
        except Exception as e:
            print(f"Error during step {steps_in_episode}: {e}")
            break

    episode_rewards.append(total_reward)
    
    # Update target network
    if episode % UPDATE_TARGET_EVERY == 0:
        target_model.load_state_dict(model.state_dict())
        print(f"Target network updated at episode {episode}")

    # Decay epsilon
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

    # Progress reporting
    if episode % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
        print(f"Episode {episode+1}/{EPISODES}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Average Reward (last 10): {avg_reward:.2f}")
        print(f"  Epsilon: {EPSILON:.3f}")
        print(f"  Steps: {steps_in_episode}")
        print(f"  Memory size: {len(memory)}")
        print("-" * 50)

    # Save model periodically
    if episode > 0 and episode % 100 == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode': episode,
            'epsilon': EPSILON,
            'episode_rewards': episode_rewards
        }, f"turtlebot_dqn_checkpoint_{episode}.pth")
        print(f"Checkpoint saved at episode {episode}")

# Save final model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'episode': EPISODES,
    'epsilon': EPSILON,
    'episode_rewards': episode_rewards
}, "turtlebot_dqn_final.pth")

print("Training complete! Final model saved as 'turtlebot_dqn_final.pth'")

# Print training statistics
print(f"\nTraining Statistics:")
print(f"Total episodes: {EPISODES}")
print(f"Final epsilon: {EPSILON:.3f}")
print(f"Average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
print(f"Best episode reward: {max(episode_rewards):.2f}")

env.close()