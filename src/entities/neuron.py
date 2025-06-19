import numpy as np
from math import exp, isfinite

class Neuron:
    """
    Neurônio com pesos reais (inclui bias).
    Responsabilidade: soma ponderada + ativação logística.
    """
    def __init__(self, weights: np.ndarray):
        if weights.ndim != 1:
            raise ValueError("weights must be a 1-D vector")
        self.weights = weights.astype(float, copy=True)

    @staticmethod
    def _safe_sigmoid(x: float) -> float:
        """Sigmoide numéricamente estável."""
        if x >= 0:
            z = exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = exp(x)
            return z / (1.0 + z)

    def forward(self, inputs: np.ndarray) -> float:
        """
        Propaga os inputs pelo neurônio.
        :param inputs: array shape=(n_inputs,)
        :return: saída ativada
        """
        if inputs.ndim != 1:
            raise ValueError("inputs must be a 1-D vector")

        if inputs.size + 1 != self.weights.size:
            raise ValueError(
                f"Input length {inputs.size} incompatible with weights "
                f"length {self.weights.size}"
            )

        z = float(np.dot(self.weights[:-1], inputs) + self.weights[-1])  # bias
        return self._safe_sigmoid(z)
