import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# ========== 0. Device Setup ==========
# 判斷 CUDA 是否可用，並設定 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== 1. Define the Environment ==========

class ADTEnv:
    def __init__(self, scores, labels, k=2, alpha=0.9, beta=0.1):
        self.scores = np.array(scores, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.int32)
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.reset()
    
    def reset(self):
        self.t = 0
        self.history = []
        return self._get_state()
    
    def _get_state(self):
        start = max(0, self.t - self.k)
        window_scores = self.scores[start:self.t]
        if len(window_scores) == 0:
            mu, sigma = 0.0, 0.0
        else:
            mu = float(np.mean(window_scores))
            sigma = float(np.var(window_scores)) if len(window_scores)>1 else 0.0
        
        window_hist = self.history[-self.k:]
        n = len(window_hist)
        tp = sum(1 for p,y in window_hist if p==1 and y==1)
        tn = sum(1 for p,y in window_hist if p==0 and y==0)
        fp = sum(1 for p,y in window_hist if p==1 and y==0)
        fn = sum(1 for p,y in window_hist if p==0 and y==1)
        if n>0:
            rho_tp, rho_tn = tp/n, tn/n
            rho_fp, rho_fn = fp/n, fn/n
        else:
            rho_tp = rho_tn = rho_fp = rho_fn = 0.0
        
        return np.array([mu, sigma, rho_tp, rho_tn, rho_fp, rho_fn], dtype=np.float32)
    
    def step(self, action):
        score = self.scores[self.t]
        true_label = self.labels[self.t]
        pred = 1 if score >= action else 0
        self.history.append((pred, int(true_label)))
        
        window = self.history[-self.k:]
        tp = sum(1 for p,y in window if p==1 and y==1)
        tn = sum(1 for p,y in window if p==0 and y==0)
        fp = sum(1 for p,y in window if p==1 and y==0)
        fn = sum(1 for p,y in window if p==0 and y==1)
        reward = self.alpha*(tp - fp - fn) + self.beta*tn
        
        self.t += 1
        done = self.t >= len(self.scores)
        next_state = self._get_state() if not done else None
        return next_state, reward, done, {}

# ========== 2. Define the DQN Agent w/ l parameter ==========

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, env, l=10, gamma=0.99, lr=1e-3,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=32, target_update=10):
        self.env = env
        self.l = l  # 每隔 l 個 step 更新一次 action
        self.state_dim = 6
        self.action_dim = 2
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        self.device = device
        self.q_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_size)
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                s = torch.from_numpy(state).unsqueeze(0).to(self.device)
                q_values = self.q_net(s)
                return int(torch.argmax(q_values[0]).item())
    
    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones   = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        non_final_mask = torch.tensor([s is not None for s in next_states], dtype=torch.bool)
        non_final_next_states = torch.tensor(
            [s for s in next_states if s is not None],
            dtype=torch.float32
        ).to(self.device)
        
        q_values = self.q_net(states).gather(1, actions)
        next_q = torch.zeros(self.batch_size, 1, device=self.device)
        next_q[non_final_mask] = (
            self.target_net(non_final_next_states).max(1)[0]
            .detach()
            .unsqueeze(1)
        )
        target = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def forward(self, state):
        return self.select_action(state)
    
    def fit(self, num_episodes=100):
        for episode in range(1, num_episodes+1):
            state = self.env.reset()
            done = False
            step_count = 0
            prev_action = self.select_action(state)  # 初始化第一個 action
            while not done:
                # 每隔 l 才重新選 action，否則沿用 prev_action
                if step_count % self.l == 0:
                    action = self.select_action(state)
                else:
                    action = prev_action
                prev_action = action

                next_state, reward, done, _ = self.env.step(action)
                self.store(state, action, reward, next_state, done)
                self.learn()
                state = next_state
                step_count += 1

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

# ========== 3. Simulate Data and Run Demo ==========

def generate_demo_sequence(length=200, anomaly_periods=[(50, 70), (140, 160)]):
    scores = np.random.normal(0.2, 0.05, size=length)
    labels = np.zeros(length, dtype=int)
    for (start, end) in anomaly_periods:
        scores[start:end] = np.random.normal(0.8, 0.05, size=end-start)
        labels[start:end] = 1
    return np.clip(scores, 0, 1), labels

scores, labels = generate_demo_sequence()
env = ADTEnv(scores, labels, k=5, alpha=0.9, beta=0.1)
# 傳入 l=10，以呼應論文最佳設定
agent = DQNAgent(env, l=10)
agent.fit(num_episodes=20000)  # training

# ========== 4. Inference & Plot ==========
env = ADTEnv(scores, labels, k=5, alpha=0.9, beta=0.1)
state = env.reset()
thresholds, preds = [], []
for t in range(len(scores)):
    if t % agent.l == 0:
        action = agent.forward(state)
    else:
        action = prev_action
    prev_action = action

    thresholds.append(action)
    next_state, _, done, _ = env.step(action)
    preds.append(int(scores[env.t-1] > action))
    if done:
        break
    state = next_state

plt.figure(figsize=(10, 5))
plt.plot(range(len(scores)), scores, label="Anomaly Score")
plt.step(range(len(thresholds)), thresholds, where='post', label="Dynamic Threshold")
for i, lbl in enumerate(labels):
    if lbl == 1:
        plt.axvspan(i, i+1, color='red', alpha=0.1)
plt.xlabel("Time")
plt.ylabel("Anomaly Score / Threshold")
plt.title("Dynamic Thresholding Demo")
plt.legend()
plt.tight_layout()
plt.show()
