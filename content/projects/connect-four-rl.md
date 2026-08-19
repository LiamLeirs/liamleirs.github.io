---
title: "RL Connect Four"
description: "Training a Connect Four agent using PPO, self-play, and an Elo-based opponent league."
tags:
  - Reinforcement Learning
  - PPO
  - Self-Play
  - Python
  - PyTorch
  - Gymnasium
draft: false
---

## Teaching an Agent to Play Connect Four

For this project, I built a reinforcement learning system that learns to play **Connect Four using Proximal Policy Optimization (PPO) and self-play**.

Instead of training against one fixed opponent, the agent competes against a league containing classical baseline agents and historical versions of itself. As the learner improves, previous versions of the policy become new opponents.

The final experiment trained the agent for **1,000,000 timesteps**.

![PPO agent playing Connect Four](/projects/connect-four/demo.gif)

---

## The Idea

Connect Four is simple to play but provides an interesting reinforcement learning problem.

The agent needs to learn concepts such as:

- creating winning combinations;
- blocking opponent threats;
- controlling useful positions;
- planning beyond the immediate move;
- adapting to different opponent strategies.

Rather than explicitly programming these strategies into the PPO agent, I wanted to investigate how much it could learn through interaction and self-play.

---

## Environment

I implemented Connect Four as a custom **Gymnasium environment**.

The environment handles:

- board state and game rules;
- legal move detection;
- alternating players;
- win and draw detection;
- reward calculation;
- randomized player order;
- varied starting positions;
- rendering for human play.

Because only non-full columns are valid actions, I used **action masking** through `MaskablePPO` from `sb3-contrib`.

This prevents the policy from selecting illegal columns instead of requiring it to learn legality through negative rewards.

---

## PPO and Self-Play

The learning agent uses **Maskable PPO**.

A major challenge with adversarial reinforcement learning is deciding **who the agent should train against**.

Training only against a weak fixed opponent can produce a policy that exploits that particular opponent without developing more general strategies.

I therefore implemented a self-play league containing:

- Random Agent
- Tactical Agent
- Minimax depth 2
- Minimax depth 4
- Historical PPO checkpoints

During training, snapshots of the PPO policy are periodically frozen and added to the league.

```text
                 ┌──────────────────┐
                 │   PPO Learner    │
                 └────────┬─────────┘
                          │
                    Training games
                          │
                          ▼
                 ┌──────────────────┐
                 │  Self-Play League│
                 └────────┬─────────┘
                          │
            ┌─────────────┼──────────────┐
            ▼             ▼              ▼
        Baselines     Minimax      Previous PPOs
```

This creates an evolving curriculum: as the learner changes, so does part of its opponent population.

---

## Elo-Based League

To estimate the relative strength of agents in the league, I implemented an **Elo rating system**.

The expected score between two agents is calculated as:

`E_A = 1 / (1 + 10^((R_B - R_A) / 400))`

where:

- `E_A` is the expected score of agent A;
- `R_A` is the Elo rating of agent A;
- `R_B` is the Elo rating of agent B.

A win is worth `1`, a draw `0.5`, and a loss `0`.

Importantly, I separated **training games from rating games**.

PPO training continuously updates the neural network, while dedicated evaluation periods determine Elo changes. Ratings for all evaluated opponents are calculated from the same frozen learner rating and applied afterwards, avoiding order-dependent rating updates.

The league rating is then also used as one signal for selecting useful training opponents.

---

## Classical Opponents

To measure progress against fixed strategies, I implemented several non-learning agents.

### Random

Selects a random legal action.

### Tactical

Uses simple tactical rules to find immediate wins and threats.

### Minimax

The strongest classical baseline uses **depth-limited Minimax with alpha-beta pruning**.

Its heuristic considers potential winning lines, opponent threats and center control.

I evaluated two versions:

- **Minimax depth 2**
- **Minimax depth 4**

These fixed opponents are particularly useful because, unlike historical PPO policies, their behaviour does not change as training progresses.

---

## Training

The final experiment ran for:

**1,000,000 PPO timesteps**

Throughout training, the current policy was periodically evaluated against members of the league.

The results were written to CSV, allowing the evolution of the policy to be analysed independently from TensorBoard training statistics.

### League Elo

![Learner Elo during self-play training](/projects/connect-four/learner_elo.png)

The learner's Elo generally increased as the policy became stronger.

However, Elo should not be interpreted as an absolute measure of Connect Four ability. Because the opponent population itself evolves, it primarily describes performance relative to the self-play league.

---

## Learning Progression

Fixed opponents provide a clearer picture of what the policy actually learned.

![Performance against fixed opponents](/projects/connect-four/baseline_performance.png)

Early in training, the PPO policy struggled even against relatively simple opponents.

As training progressed, it became substantially stronger against the Random and Tactical agents and eventually became competitive with the Minimax agents.

### Minimax Performance

![Performance against Minimax](/projects/connect-four/minimax_performance.png)

One interesting result was the difference between the two Minimax depths.

The PPO agent eventually learned strategies that performed strongly against the shallower Minimax agent, while **Minimax depth 4 remained a substantially more difficult benchmark**.

Performance against the stronger opponent eventually approached roughly competitive levels rather than continuing to improve indefinitely.

---

## Did Self-Play Keep Improving the Agent?

Looking only at fixed opponents does not tell the entire story.

I also evaluated newer policies against historical PPO checkpoints.

![Performance against historical PPO checkpoints](/projects/connect-four/historical_ppo_performance.png)

Later policies frequently performed very strongly against previous versions of themselves.

Interestingly, this improvement did not always translate into equally large gains against Minimax.

This illustrates an important limitation of self-play: **becoming better against the evolving training population does not necessarily mean becoming universally better at the game.**

---

## What Didn't Work Perfectly

The final policy does **not solve Connect Four**.

Despite strong evaluation results against parts of the training league, manual games revealed that the policy can still make surprisingly poor tactical decisions, including occasionally failing to respond correctly to immediate threats.

This was an important result in itself.

Aggregate win rates and Elo ratings can make a policy appear stronger than it actually is. A policy may learn strategies that work well against particular opponents without learning every fundamental concept of the game robustly.

Varied starting states were therefore added to evaluation to reduce dependence on repeatedly encountering similar trajectories from an empty board.

---

## What I Learned

This project ended up being about much more than simply training PPO.

I gained practical experience with:

- designing a custom Gymnasium environment;
- reinforcement learning with PPO;
- legal-action masking;
- adversarial self-play;
- opponent sampling and curriculum design;
- Elo rating systems;
- Minimax and alpha-beta pruning;
- reproducible evaluation;
- checkpoint management;
- experiment logging and visualization;
- automated testing.

The biggest lesson was that **evaluating reinforcement learning agents is often just as important as training them**.

A rising reward or Elo rating alone does not prove that a policy has learned the behaviour you expect. Fixed baselines, historical policies, varied states and manual inspection can reveal very different aspects of performance.

---

## Future Work

There are several directions in which the project could be extended:

- improved opponent sampling;
- vectorized environments;
- hyperparameter optimization;
- larger policy networks;
- tactical state evaluation;
- deeper Minimax opponents;
- checkpoint tournaments;
- alternative rating systems such as Glicko or TrueSkill;
- DQN-based agents;
- Monte Carlo Tree Search;
- AlphaZero-style policy/value learning.

For this project, however, I intentionally kept the final version focused on **PPO and self-play** rather than continually expanding the system.

---

## Technologies

`Python` · `PyTorch` · `Gymnasium` · `Stable-Baselines3` · `sb3-contrib` · `PPO` · `NumPy` · `Pandas` · `Matplotlib` · `Pytest`

---

## Source Code

The complete implementation, training code, evaluation tools and tests are available on GitHub.

**[View the project on GitHub](https://github.com/LiamLeirs/RL-Connect-Four)**
