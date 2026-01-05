# q1_dice_blackjack_qlearning.py
# Dice Blackjack (discrete die) + Q-learning with epsilon-greedy

import random


# -------------------------
# Environment: Dice Blackjack
# -------------------------
HIT = 0
STAY = 1

MAX_SAFE_SUM = 11

def step(state_s: int, action: int):
    """
    One environment step.
    Returns: next_state, reward, done
    """
    if action == STAY:
        # Staying ends the episode with reward equal to current sum (<= 11)
        return state_s, state_s, True

    # HIT: roll a fair 6-sided die
    x = random.randint(1, 6)
    new_s = state_s + x

    if new_s > MAX_SAFE_SUM:
        # Bust
        return new_s, 0, True

    # Not terminal yet
    return new_s, 0, False


# -------------------------
# Q-learning agent
# -------------------------
def epsilon_greedy_action(q_table, s: int, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.choice([HIT, STAY])
    # Greedy action (tie-break randomly)
    q_hit = q_table[s][HIT]
    q_stay = q_table[s][STAY]
    if q_hit > q_stay:
        return HIT
    if q_stay > q_hit:
        return STAY
    return random.choice([HIT, STAY])


def train_q_learning(
    episodes: int = 200_000,
    alpha: float = 0.1,
    gamma: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
):
    # States we care about in Q-table: sums 0..11
    # (If we bust, episode ends immediately; no need to store bust states.)
    q = [[0.0, 0.0] for _ in range(MAX_SAFE_SUM + 1)]

    for ep in range(episodes):
        # Linear epsilon decay
        t = ep / max(1, episodes - 1)
        epsilon = epsilon_start + t * (epsilon_end - epsilon_start)

        s = 0
        done = False

        while not done:
            a = epsilon_greedy_action(q, s, epsilon)
            s_next, r, done = step(s, a)

            # Q-learning update
            target = r
            if not done:
                # s_next is guaranteed <= 11 here
                target += gamma * max(q[s_next][HIT], q[s_next][STAY])

            q[s][a] = q[s][a] + alpha * (target - q[s][a])

            # Advance state if not terminal
            if not done:
                s = s_next

    return q


def evaluate_policy(q_table, episodes: int = 100) -> float:
    total = 0.0
    for _ in range(episodes):
        s = 0
        done = False
        while not done:
            # Greedy policy (tie-break randomly)
            q_hit = q_table[s][HIT]
            q_stay = q_table[s][STAY]
            if q_hit > q_stay:
                a = HIT
            elif q_stay > q_hit:
                a = STAY
            else:
                a = random.choice([HIT, STAY])

            s, r, done = step(s, a)
            if done:
                total += r
    return total / episodes


def describe_learned_policy(q_table):
    """
    Prints whether policy is roughly threshold-based:
    i.e., HIT below some sum and STAY from some sum upward.
    """
    actions = []
    for s in range(MAX_SAFE_SUM + 1):
        q_hit = q_table[s][HIT]
        q_stay = q_table[s][STAY]
        if q_hit > q_stay:
            actions.append("H")
        elif q_stay > q_hit:
            actions.append("S")
        else:
            actions.append("?")  # tie

    # Find first index where action is S and remains S (ignoring ties)
    threshold = None
    for s in range(MAX_SAFE_SUM + 1):
        if actions[s] == "S":
            # check if from here onward it's mostly stay (S or ties)
            if all(a in ("S", "?") for a in actions[s:]):
                threshold = s
                break

    print("Greedy actions by sum S=0..11 (H=hit, S=stay, ?=tie):")
    print("".join(actions))
    if threshold is not None:
        print(f"Approx. threshold strategy: tends to STAY from S >= {threshold}")
    else:
        print("Policy is not a clean single-threshold (may mix actions or ties).")


if __name__ == "__main__":
    random.seed(0)  # for reproducibility

    q = train_q_learning(
        episodes=200_000,
        alpha=0.1,
        gamma=1.0,
        epsilon_start=1.0,
        epsilon_end=0.05,
    )

    avg = evaluate_policy(q, episodes=100)
    print(f"Average reward over 100 evaluation episodes: {avg:.3f}")
    describe_learned_policy(q)
