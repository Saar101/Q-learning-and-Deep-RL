# q2_dice_blackjack_continuous_mc.py
# Continuous Dice Blackjack + Monte Carlo learning for Q(S, hit)

import random
import torch
import torch.nn as nn
import torch.optim as optim


MAX_SAFE_SUM = 11.0


# -------------------------
# Environment step
# -------------------------
def step(state_s: float, action: str):
    """
    action: "hit" or "stay"
    returns: next_state, reward, done
    """
    if action == "stay":
        return state_s, state_s, True

    # hit: sample uniformly from [1, 6]
    x = random.uniform(1.0, 6.0)
    new_s = state_s + x

    if new_s > MAX_SAFE_SUM:
        return new_s, 0.0, True

    return new_s, 0.0, False


# -------------------------
# Q-network: Q(S, hit)
# -------------------------
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, s):
        return self.net(s)


# -------------------------
# Training
# -------------------------
def train(
    episodes=50_000,
    epsilon_start=1.0,
    epsilon_end=0.05,
    lr=1e-3
):
    qnet = QNet()
    optimizer = optim.Adam(qnet.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(episodes):
        # linear epsilon decay
        t = ep / max(1, episodes - 1)
        epsilon = epsilon_start + t * (epsilon_end - epsilon_start)

        s = 0.0
        done = False

        # store states where we took HIT
        hit_states = []

        while not done:
            s_tensor = torch.tensor([[s / MAX_SAFE_SUM]], dtype=torch.float32)
            q_hit = qnet(s_tensor).item()
            stay_value = s

            if random.random() < epsilon:
                action = random.choice(["hit", "stay"])
            else:
                action = "hit" if q_hit > stay_value else "stay"

            if action == "hit":
                hit_states.append(s)

            s, reward, done = step(s, action)

        # Monte Carlo update: final reward for all hit states
        if hit_states:
            targets = torch.tensor([[reward]] * len(hit_states), dtype=torch.float32)
            inputs = torch.tensor(
                [[hs / MAX_SAFE_SUM] for hs in hit_states],
                dtype=torch.float32
            )

            preds = qnet(inputs)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return qnet


# -------------------------
# Evaluation
# -------------------------
def evaluate(qnet, episodes=100):
    total = 0.0
    for _ in range(episodes):
        s = 0.0
        done = False
        while not done:
            s_tensor = torch.tensor([[s / MAX_SAFE_SUM]], dtype=torch.float32)
            q_hit = qnet(s_tensor).item()
            action = "hit" if q_hit > s else "stay"
            s, reward, done = step(s, action)
            if done:
                total += reward
    return total / episodes


# -------------------------
# Describe learned threshold
# -------------------------
def estimate_threshold(qnet):
    for s in [i * 0.1 for i in range(0, 111)]:
        s_tensor = torch.tensor([[s / MAX_SAFE_SUM]], dtype=torch.float32)
        if qnet(s_tensor).item() <= s:
            return round(s, 2)
    return None


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    qnet = train()
    avg = evaluate(qnet)

    threshold = estimate_threshold(qnet)

    print(f"Average reward over 100 evaluation episodes: {avg:.3f}")
    print(f"Approximate learned threshold for staying: S ≈ {threshold}")

# Evaluation: Over 100 evaluation episodes (greedy policy), the learned agent achieved an average reward of 7.970.
# Learned policy / threshold: The policy is approximately threshold-based: it tends to Hit for smaller sums and Stay once the current sum reaches about 𝑆≈ 6.6 .
# Learned Q-function: The learned 𝑄(𝑆,hit) decreases as 𝑆 increases, reflecting the rising probability of busting when hitting from larger sums. The agent stops when 𝑄(𝑆,hit) drops below the deterministic value of staying (which equals 𝑆).