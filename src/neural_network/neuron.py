from typing import List
import numpy as np

class Neuron:
    def __init__(self, weights: List[float]) -> None:
        if len(weights) != 10:
            raise ValueError("Neuron must have exactly 10 weights.")
        self.weights = weights

    def activate(self, v: float, function: str = "relu") -> float:
        if function == "relu":
            return max(0.0, v)
        elif function == "sigmoid":
            return 1 / (1 + np.exp(-v))
        else:
            raise ValueError(f"Unsupported activation function: {function}")

    def propagate(self, inputs: List[float], activation: str = "relu") -> float:
        if len(inputs) != 9:
            raise ValueError("Input must have exactly 9 values.")
        extended_input = [1.0] + inputs
        v = sum(i * w for i, w in zip(extended_input, self.weights))
        return self.activate(v, activation)
