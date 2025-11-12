# ==========================================================================
# Judul  : Agen Taksi Cerdas untuk Pengantaran Penumpang di Kota Mini
# Nama   : Intan Telaumbanua
# NIM    : 301230016
# Kelas  : IF 5B
# ==========================================================================
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import colors
import os

# ===============================================================
# 1. ENVIRONMENT TAXI 5x5
# ===============================================================
class TaxiEnv:
    def __init__(self, grid_size=5, pickup_points=None, seed=None):
        self.grid_size = grid_size
        if pickup_points is None:
            self.pickup_points = [(0,0),(0,grid_size-1),(grid_size-1,0),(grid_size-1,grid_size-1)]
        else:
            self.pickup_points = pickup_points
        self.n_pick = len(self.pickup_points)
        self.action_space = 6  # 0:Up,1:Down,2:Right,3:Left,4:Pickup,5:Dropoff
        self.reset(seed=seed)

        # buat daftar semua kombinasi state
        self.state_space = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                for p in range(self.n_pick + 1):
                    for d in range(self.n_pick):
                        self.state_space.append((r, c, p, d))
        self.num_states = len(self.state_space)
        self.num_actions = self.action_space

    def reset(self, random_start=True, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.taxi_row = random.randint(0, self.grid_size-1)
        self.taxi_col = random.randint(0, self.grid_size-1)
        self.pass_idx = random.randint(0, self.n_pick-1)
        self.dest_idx = random.randint(0, self.n_pick-1)
        while self.dest_idx == self.pass_idx:
            self.dest_idx = random.randint(0, self.n_pick-1)
        self.done = False
        return self.encode()

    def encode(self):
        i = self.taxi_row
        i = i * self.grid_size + self.taxi_col
        i = i * (self.n_pick + 1) + self.pass_idx
        i = i * self.n_pick + self.dest_idx
        return i

    def decode(self, idx):
        dest_idx = idx % self.n_pick
        idx //= self.n_pick
        pass_idx = idx % (self.n_pick + 1)
        idx //= (self.n_pick + 1)
        taxi_col = idx % self.grid_size
        taxi_row = idx // self.grid_size
        return (taxi_row, taxi_col, pass_idx, dest_idx)

    def step(self, action):
        reward = -1
        info = {'invalid': False}
        if action == 0 and self.taxi_row > 0: self.taxi_row -= 1
        elif action == 1 and self.taxi_row < self.grid_size-1: self.taxi_row += 1
        elif action == 2 and self.taxi_col < self.grid_size-1: self.taxi_col += 1
        elif action == 3 and self.taxi_col > 0: self.taxi_col -= 1
        elif action == 4:
            if self.pass_idx < self.n_pick and (self.taxi_row, self.taxi_col) == self.pickup_points[self.pass_idx]:
                self.pass_idx = self.n_pick
            else:
                reward -= 5; info['invalid'] = True
        elif action == 5:
            if self.pass_idx == self.n_pick and (self.taxi_row, self.taxi_col) == self.pickup_points[self.dest_idx]:
                reward += 20; self.done = True
            elif self.pass_idx == self.n_pick:
                reward -= 10; self.done = True
            else:
                reward -= 5; info['invalid'] = True
        next_state = self.encode()
        return next_state, reward, self.done, info

# ===============================================================
# 2. TRAINING DENGAN Q-LEARNING
# ===============================================================
def train_q_learning(env, episodes=8000, alpha=0.8, gamma=0.9,
                     epsilon=1.0, eps_decay=0.9996, eps_min=0.05,
                     max_steps_per_episode=200, seed=None, verbose=False):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    Q = np.zeros((env.num_states, env.num_actions))
    rewards_per_episode = []
    eps = epsilon
    for ep in range(episodes):
        state = env.reset(random_start=True)
        total_reward = 0
        done = False
        steps = 0
        while not done and steps < max_steps_per_episode:
            action = random.randint(0, env.num_actions-1) if random.random() < eps else np.argmax(Q[state])
            next_state, reward, done, _ = env.step(action)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
            total_reward += reward
            steps += 1
        eps = max(eps*eps_decay, eps_min)
        rewards_per_episode.append(total_reward)
        if verbose and (ep+1) % (episodes//5) == 0:
            print(f"Episode {ep+1}/{episodes} | reward={total_reward:.1f} | eps={eps:.3f}")
    return Q, rewards_per_episode

# ===============================================================
# 3. VISUALISASI
# ===============================================================
def moving_average(data, window=100):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_learning_curve(rewards, window=100, title="Learning Curve", save_path=None):
    ma = moving_average(rewards, window)
    plt.figure(figsize=(10,4))
    plt.plot(rewards, alpha=0.3, label="Episode Reward")
    plt.plot(np.arange(len(ma))+window-1, ma, label=f"Moving Avg (window={window})", linewidth=2)
    plt.title(title); plt.xlabel("Episode"); plt.ylabel("Reward"); plt.legend(); plt.grid(True)
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

def visualize_policy_grid(env, Q, title="Policy Map", save_path=None):
    actions = np.argmax(Q, axis=1)
    grid = np.zeros((env.grid_size, env.grid_size), dtype=int)
    symbols = {0:'↑',1:'↓',2:'→',3:'←',4:'P',5:'D'}
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            idx = env.state_space.index((r, c, 0, 1))
            grid[r, c] = actions[idx]
    cmap = colors.ListedColormap(['#3b3b98','#e1b12c','#4cd137','#8c7ae6','#44bd32','#e84118'])
    norm = colors.BoundaryNorm(np.arange(-0.5,6.5,1), cmap.N)
    plt.figure(figsize=(6,6))
    plt.imshow(grid, cmap=cmap, norm=norm)
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            plt.text(j, i, symbols[grid[i,j]], ha='center', va='center', color='white', fontsize=16, fontweight='bold')
    plt.title(title)
    cbar = plt.colorbar(ticks=range(6))
    cbar.set_ticklabels(['Up','Down','Right','Left','Pickup','Dropoff'])
    if save_path: plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

# ===============================================================
# 4. EKSEKUSI UTAMA
# ===============================================================
if __name__ == "__main__":
    os.makedirs("ta07_outputs", exist_ok=True)
    env = TaxiEnv(grid_size=5, seed=42)

    # ---- Training utama ----
    Q, rewards = train_q_learning(env, episodes=8000, gamma=0.9, seed=42, verbose=True)
    plot_learning_curve(rewards, window=100, title="Learning Curve (Gamma=0.9)",
                        save_path="ta07_outputs/learning_curve_gamma09.png")
    visualize_policy_grid(env, Q, title="Policy Map (Gamma=0.9)",
                          save_path="ta07_outputs/policy_gamma09.png")

    # ---- Eksperimen Gamma ----
    Q_low, r_low = train_q_learning(env, episodes=8000, gamma=0.1, seed=42)
    ma_high = moving_average(rewards, window=100)
    ma_low = moving_average(r_low, window=100)

    plt.figure(figsize=(10,4))
    plt.plot(ma_high, label="Gamma=0.9")
    plt.plot(ma_low, label="Gamma=0.1")
    plt.title("Perbandingan Dampak Nilai Gamma terhadap Reward")
    plt.xlabel("Episode"); plt.ylabel("Moving Avg Reward"); plt.legend(); plt.grid(True)
    plt.savefig("ta07_outputs/compare_gamma.png", dpi=200, bbox_inches='tight')
    plt.show()

    visualize_policy_grid(env, Q_low, title="Policy Map (Gamma=0.1)",
                          save_path="ta07_outputs/policy_gamma01.png")

    # ---- Simpulan otomatis ----
    print("\n=== RINGKASAN HASIL ===")
    print(f"Total reward akhir (γ=0.9): {np.mean(rewards[-100:]):.2f}")
    print(f"Total reward akhir (γ=0.1): {np.mean(r_low[-100:]):.2f}")
    print("Policy map dan grafik disimpan di folder: ta07_outputs/")
