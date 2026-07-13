# Reinforcement Learning Final Project - MCTS & AlphaZero-Style Self-Play

**Final Project (M.Sc. Data Science, HIT). Monte Carlo Tree Search implemented from scratch for two-player board games — Tic-Tac-Toe as the exactly-solved verification environment, a compact 4×5 Connect-Four as the main one — then extended into an AlphaZero-style self-play agent: a policy-value network guiding PUCT search in place of random rollouts.**

## Headline Results
- **Tic-Tac-Toe, equal 50-simulation budget:** network-guided search beats pure MCTS **18–0** head-to-head (22 draws), cuts losses vs. a perfect player from **21/60 to 5/60**, and lifts the optimal-move rate from 0.88 to 1.00.
- **Tic-Tac-Toe, larger budgets:** pure MCTS needs 1,000 simulations to match the network's losses at 200 (3/60 vs. 2/60).
- **Connect-Four, equal 60-simulation budget:** network-guided search beats pure MCTS **13–7** head-to-head.

## Key Features
- **Pure MCTS from Scratch:** Full UCT implementation — selection, expansion, rollout, backpropagation — validated against a memoized minimax solver that provides a perfect opponent and an exact optimal-move oracle.
- **AlphaZero-Style Self-Play:** A two-headed policy-value network guiding PUCT search with priors, Dirichlet root noise, a rolling replay buffer, and a combined cross-entropy + MSE loss — trained purely from self-play games.
- **Ablations:** Simulation-budget and rollout-depth studies on Connect-Four (win rate rising from 0.43 at depth 1 to ~0.76 at depth 8+).
- **Rigor & Visualization:** Fixed seeds, symmetric two-sided evaluation, value-head calibration against 4,520 minimax ground-truth states, and a frame-by-frame rendered Connect-Four game.

## Repository Structure
- `final_project_reinforcement_learning.ipynb`: Full solution notebook — environments, pure MCTS, minimax validation, the AlphaZero-style training loop, ablations, and analysis.
- `Final_Project_Instructions.pdf`: Original course project guidelines.
