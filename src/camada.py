from neuronio import neuronio  # Import the neuronio class if it's in another file

class CamadaSaida:
    def __init__(sel, neuronios: neuronio):
        """
        Initialize an output layer with 9 neurons.
        """
        self.neuronios = neuronios
    
    def set_neuron(self, index, value):
        """Set the value of a specific neuron."""
        if 0 <= index < 9:
            self.Neuronios[index] = value
        else:
            raise IndexError("Neuron index must be between 0-8")
    
    def get_neuron(self, index):
        """Get the value of a specific neuron."""
        if 0 <= index < len(self.Neuronios):
            return self.Neuronios[index]
        else:
            raise IndexError("Neuron index must be between 0-8")
    
    def get_all_neurons(self):
        """Return all neuron values."""
        return self.Neuronios