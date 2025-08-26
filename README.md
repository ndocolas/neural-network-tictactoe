# Neural Network – Tic-Tac-Toe (from scratch)

This project runs a neural network truly from scratch: every piece is hand-coded — the neuron, the layers, the MLP itself, the chromosome representation, the Genetic Algorithm, and the Minimax engine. No high-level frameworks, no shortcuts: we implemented the math, the evolutionary loop, and the game logic by hand to train an AI that learns to play Tic-Tac-Toe against a Minimax opponent (with and without randomness). The result is a compact, hands-on lab where you can watch the network come to life, evolve, and play, with a simple interface to train and test.

Demo: https://www.youtube.com/watch?v=Ea-CpFXAJww

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
src/
├─ adapters/ - Integration layer that wires the core logic to opponents and training/play flows (e.g., wraps Minimax for use by 
├─ entities/ - Domain models and NN building blocks (neuron, layer, network, chromosome) as plain data/logic types.
├─ minimax/ - Minimax search engine and its variants used as opponents during training and play.
├─ services/ - Game-level services such as Tic-Tac-Toe simulation and other stateless operations.
├─ ui/ - Simple user interface to start training, monitor progress, and play matches.
├─ usecases/ - Orchestration of workflows: training loops, evaluation routines, and gameplay modes.
└─ utils/ - Shared helpers and small utilities used across the project.
```
---

## License
This project is open-source. See the [LICENSE](LICENSE) file for more details.
