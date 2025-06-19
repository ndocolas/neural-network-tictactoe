import numpy as np
from typing import Optional

class Chromosome:
    """
    Armazena vetor de pesos, score e um ID único.
    O ID é atribuído automaticamente, mas pode ser
    reaproveitado se desejado (p.ex. clonagem fiel).
    """
    _next_id: int = 0

    def __init__(self,
                 weights_vector: np.ndarray,
                 uid: Optional[int] = None):
        self.id: int = (uid if uid is not None
                        else Chromosome._next_id)
        if uid is None:
            Chromosome._next_id += 1

        self.weights_vector: np.ndarray = weights_vector
        self.score: float = 0.0

    # utilitário opcional, facilita cópias mantendo ou não o id
    def clone(self, keep_id: bool = False) -> "Chromosome":
        return Chromosome(self.weights_vector.copy(),
                          uid=self.id if keep_id else None)

    def set_score(self, score: float):
        self.score = score

    def __repr__(self):
        return f"Chromosome id={self.id} fit={self.score:.2f}"
