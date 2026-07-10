# Reinforcement Learning Final Project - MCTS & AlphaZero-Style Self-Play

**Final Project (M.Sc. Data Science, HIT). Monte Carlo Tree Search implemented from scratch for two-player board games, then extended into an AlphaZero-style self-play agent — a policy-value network guiding PUCT search. Validated against an exact minimax solver on Tic-Tac-Toe: at an equal simulation budget, the network-guided search cuts losses against perfect play from 21/60 to 5/60 and beats pure MCTS head-to-head.**

## Overview
The project follows a two-stage design on two environments sharing a unified five-function interface. Tic-Tac-Toe serves as the verification environment — small enough to solve exactly with memoized minimax, so every claim is measured against ground truth (losses vs a perfect player, share of provably-optimal moves). A compact 4×5 Connect-Four is the main environment. Stage 1 implements pure MCTS (UCT selection, expansion, random rollouts, backpropagation) and validates it; Stage 2 replaces random rollouts with a small PyTorch policy-value network trained purely from self-play, AlphaZero-style, and shows the same search budget goes much further with a learned model behind it.

## Key Features
- **Pure MCTS from Scratch:** Full UCT implementation — selection, expansion, rollout, backpropagation — wrapped as a plug-in player function compatible with every other agent in the notebook.
- **Ground-Truth Validation:** A memoized minimax solver provides a perfect opponent and an exact optimal-move oracle; pure MCTS losses vs perfect play fall from 21/60 at 50 simulations to 3/60 at 1,000, with an optimal-move rate reaching 96%.
- **AlphaZero-Style Self-Play:** A two-headed policy-value network guiding PUCT search with priors, Dirichlet root noise, a rolling replay buffer, and a combined cross-entropy + MSE loss — trained from self-play games only.
- **Headline Result:** At an equal 50-simulation budget, the network-guided search cuts losses against the perfect player from 21/60 to 5/60 and lifts the optimal-move rate from 0.88 to 1.00; on Connect-Four it beats pure MCTS 13-7 head-to-head at equal compute.
- **Ablations:** Simulation-budget and truncated-rollout-depth studies (win rate rising from 0.43 at depth 1 to ~0.76 at depth 8+), plus the learned agent against larger pure-search budgets.
- **Value-Head Calibration:** Network value predictions compared against exact minimax values over 4,520 non-terminal states — echoing the value-comparison methodology of the course homework.
- **Experimental Rigor:** Fixed seeds, symmetric two-sided evaluation (each agent plays both as opener and second player), and quantitative baselines — random and greedy players with immediate win/block logic.
- **Visualization:** Root-visit policy distributions, learned-policy heatmaps for opening positions, training curves, and a frame-by-frame rendered Connect-Four game as a bonus.

## Tech Stack
- **Language:** Python
- **Core:** NumPy; PyTorch (policy-value network) — all RL components implemented from scratch, no RL libraries (per course rules)
- **Visualization:** Matplotlib
- **Environment:** Google Colab

## Repository Structure
- `final_project_reinforcement_learning.ipynb`: Full solution notebook — environments, pure MCTS, minimax validation, the AlphaZero-style training loop, ablations, and analysis.
- `Final_Project_Instructions.pdf`: Original course project guidelines.
