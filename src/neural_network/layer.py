from typing import List
from .neuron import Neuron

class Layer:
    def __init__(self, neurons: List[Neuron]) -> None:
        self.neurons = neurons

    @classmethod
    def from_weights(cls, flat_weights: List[float], num_neurons: int) -> 'Layer':
        neurons = []
        for i in range(num_neurons):
            start = i * 10
            end = start + 10
            neurons.append(Neuron(flat_weights[start:end]))
        return cls(neurons)

    def propagate(self, inputs: List[float], activation: str) -> List[float]:
        return [neuron.propagate(inputs, activation) for neuron in self.neurons]
