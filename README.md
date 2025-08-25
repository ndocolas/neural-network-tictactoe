# Neural Network – Tic-Tac-Toe (from scratch)

This project runs a neural network truly from scratch: every piece is hand-coded — the neuron, the layers, the MLP itself, the chromosome representation, the Genetic Algorithm, and the Minimax engine. No high-level frameworks, no shortcuts: we implemented the math, the evolutionary loop, and the game logic by hand to train an AI that learns to play Tic-Tac-Toe against a Minimax opponent (with and without randomness). The result is a compact, hands-on lab where you can watch the network come to life, evolve, and play, with a simple interface to train and test.

- 🎥 Demo: https://www.youtube.com/watch?v=Ea-CpFXAJww

---

## What is this project?

- **Goal:** Evolve the weights of a 9-9-9 MLP so it plays Tic-Tac-Toe reliably.
- **Training:** Genetic Algorithm (selection, crossover, mutation, elitism).
- **Opponents:** Minimax **100%** (optimal; random first move) and **50%** (mixed random/optimal).
- **UI:** Start/monitor training and play vs the trained network.

---

## Quick Start

### Prerequisites
- Python **3.10+**
- `pip` (and optionally a virtual environment)

### Install
```bash
pip install -r requirements.txt
```

### Run (UI)
```bash
python gui.py
```
- From the menu you can:
  - Train (choose population, generations, games/individual)
  - Play: **User vs Minimax** or **User vs Trained Network**

### Run (CLI)
```bash
python main.py
```
- Prints training/evaluation steps to the terminal.
- Uses the same core logic as the UI.

> Trained weights are saved/loaded automatically by the app.

---

## Project Structure (high level)

```
/Entities        # board, move, individual/chromosome, etc.
 /Services       # neural net math, GA operators, evaluation
 /UseCases       # training loops, evaluation flows, gameplay modes
 /Adapters       # Minimax (100% and 50%) implementations
 gui.py          # simple UI (train/play)
 main.py         # CLI entry (train/evaluate)
```

---

## Results (summary)

- Learns stable play; draws consistently vs **optimal Minimax**.
- Competitive vs **mixed Minimax** depending on training settings.

---
