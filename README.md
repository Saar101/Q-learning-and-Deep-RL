# Q-learning and Deep Reinforcement Learning  
**Dice Blackjack – Discrete and Continuous Versions**

This repository contains solutions for **Exercise 4: Q-learning and Deep RL**, focusing on a simplified Blackjack variant called **Dice Blackjack**.  
The project demonstrates both **tabular reinforcement learning** and **function approximation using neural networks**, highlighting how similar decision strategies emerge in discrete and continuous state spaces.

---

## 📌 Project Overview

The Dice Blackjack game starts with an accumulated sum \(S = 0\).  
At each step, the agent can either:
- **Hit** – roll a die and add the result to the sum  
- **Stay** – stop and receive a reward equal to the current sum  

If the sum exceeds 11, the agent busts and receives a reward of 0.

The project is divided into two parts:

1. **Discrete Dice Blackjack** – solved using **Q-learning with a Q-table**  
2. **Continuous Dice Blackjack** – solved using **Monte Carlo learning with a neural network**

---

## 📂 Repository Structure

```
.
├── q1_dice_blackjack_qlearning.py
├── q2_dice_blackjack_continuous_mc.py
└── README.md
```

---

## 🧠 Question 1 – Discrete Dice Blackjack (Q-learning)

- **State space:** Discrete sums \(S \in \{0,1,\dots,11\}\)
- **Method:** Tabular Q-learning
- **Policy:** ε-greedy
- **Evaluation:** 100 episodes using a greedy policy

### Result
- **Average reward:** ~7.48  
- **Learned behavior:**  
  The agent learns a **threshold-based strategy**, choosing **Hit** for sums below 6 and **Stay** from approximately 6 and above.

---

## 🧠 Question 2 – Continuous Dice Blackjack (Deep RL)

- **State space:** Continuous \(S \in [0, \infty)\)
- **Method:** Monte Carlo learning with function approximation
- **Model:** Small fully connected neural network estimating \(Q(S,\text{hit})\)
- **Framework:** PyTorch
- **Policy:** ε-greedy comparison between \(Q(S,\text{hit})\) and the deterministic stay value \(S\)

### Result
- **Average reward:** ~7.97  
- **Learned behavior:**  
  The agent again learns an **approximate threshold strategy**, switching to **Stay** around \(S \approx 6.6\).

---

## 🔍 Key Insights

- Both discrete and continuous versions converge to **similar threshold-based policies**
- The structure of the optimal strategy is **robust to state space discretization**
- Function approximation successfully replaces tabular methods in continuous settings

---

## ▶️ How to Run

### Question 1 (Q-learning)
```bash
python q1_dice_blackjack_qlearning.py
```

### Question 2 (Deep RL – requires PyTorch)
```bash
pip install torch
python q2_dice_blackjack_continuous_mc.py
```

---

## 🎓 Academic Context

This project was completed as part of a university assignment on **Reinforcement Learning**, covering:
- Q-learning
- ε-greedy exploration
- Monte Carlo updates
- Neural network function approximation
