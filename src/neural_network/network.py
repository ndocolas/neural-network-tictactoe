from typing import List
from .layer import Layer

class NeuralNetwork:
    def __init__(self, hidden_layer: Layer, output_layer: Layer):
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer

    @classmethod
    def from_chromosome(cls, chromosome: List[float]) -> 'NeuralNetwork':
        hidden = Layer.from_weights(chromosome[:90], 9)
        output = Layer.from_weights(chromosome[90:], 9)
        return cls(hidden, output)

    def forward(self, inputs: List[float]) -> List[float]:
        hidden_output = self.hidden_layer.propagate(inputs, activation="relu")
        return self.output_layer.propagate(hidden_output, activation="sigmoid")
