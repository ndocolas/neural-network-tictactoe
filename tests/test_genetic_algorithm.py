import pytest
import numpy as np
from src.usecases.genetic_algorithm import GeneticAlgorithm
from src.entities.neural_network import NeuralNetwork

class MockEvaluator:
    """Avaliador mock para testes do algoritmo genético."""
    
    def evaluate(self, individual):
        """Avalia um indivíduo retornando uma pontuação aleatória.
        
        Para testes, usamos a soma dos valores absolutos dos pesos.
        """
        return np.sum(np.abs(individual))

def test_genetic_algorithm_initialization():
    """Testa a inicialização do algoritmo genético."""
    pop_size = 10
    input_size = 9
    hidden_size = 9
    output_size = 9
    evaluator = MockEvaluator()
    
    ga = GeneticAlgorithm(pop_size, input_size, hidden_size, output_size, evaluator)
    
    # Verifica se a população foi criada com o tamanho correto
    assert len(ga.population) == pop_size
    
    # Verifica se cada indivíduo tem o tamanho correto
    for individual in ga.population:
        expected_size = input_size * hidden_size + hidden_size * output_size + hidden_size + output_size
        assert len(individual.weights) == expected_size

def test_selection():
    """Testa o método de seleção por torneio."""
    pop_size = 10
    input_size = 9
    hidden_size = 9
    output_size = 9
    evaluator = MockEvaluator()
    
    ga = GeneticAlgorithm(pop_size, input_size, hidden_size, output_size, evaluator)
    
    # Atribui pontuações conhecidas para teste
    for i, individual in enumerate(ga.population):
        individual.fitness = i  # Indivíduos com índices maiores têm melhor fitness
    
    # Seleciona os melhores
    selected = ga._tournament_selection(k=3, tournament_size=5)
    
    # Verifica se os selecionados estão entre os melhores
    min_selected = min(individual.fitness for individual in selected)
    assert min_selected >= pop_size - 3  # Os 3 melhores devem ter fitness >= 7

def test_crossover():
    """Testa o operador de crossover."""
    input_size = 9
    hidden_size = 9
    output_size = 9
    
    # Cria dois pais com pesos conhecidos
    parent1 = np.ones(input_size * hidden_size + hidden_size * output_size + hidden_size + output_size)
    parent2 = np.zeros_like(parent1)
    
    # Aplica o crossover
    child1, child2 = GeneticAlgorithm._crossover(parent1, parent2)
    
    # Verifica se os filhos têm o mesmo tamanho que os pais
    assert len(child1) == len(parent1)
    assert len(child2) == len(parent2)
    
    # Verifica se os filhos são diferentes dos pais (a menos que sejam iguais por acaso)
    assert not np.array_equal(child1, parent1) or not np.array_equal(child1, parent2)
    assert not np.array_equal(child2, parent1) or not np.array_equal(child2, parent2)

def test_mutation():
    """Testa o operador de mutação."""
    # Cria um indivíduo com valores conhecidos
    individual = np.zeros(100)
    mutation_rate = 0.1
    
    # Aplica a mutação
    mutated = GeneticAlgorithm._mutate(individual, mutation_rate)
    
    # Verifica se alguns valores foram alterados (com alta probabilidade)
    assert not np.array_equal(individual, mutated)
    
    # Verifica se aproximadamente 10% dos genes foram mutados (com tolerância)
    num_mutations = np.sum(individual != mutated)
    assert 5 <= num_mutations <= 15  # Esperado em torno de 10

def test_evolve():
    """Testa uma execução completa do algoritmo genético."""
    pop_size = 10
    input_size = 9
    hidden_size = 9
    output_size = 9
    evaluator = MockEvaluator()
    
    ga = GeneticAlgorithm(pop_size, input_size, hidden_size, output_size, evaluator)
    
    # Executa algumas gerações
    best_individual, best_fitness = ga.evolve(generations=5, games_per_gen=1, verbose=False)
    
    # Verifica se o melhor indivíduo tem uma pontuação válida
    assert best_fitness >= 0
    assert len(best_individual) == input_size * hidden_size + hidden_size * output_size + hidden_size + output_size

def test_create_neural_network():
    """Testa a criação de uma rede neural a partir de um indivíduo."""
    input_size = 9
    hidden_size = 9
    output_size = 9
    
    # Cria um indivíduo com pesos conhecidos
    weights = np.arange(input_size * hidden_size + hidden_size * output_size + hidden_size + output_size)
    
    # Cria a rede neural
    nn = GeneticAlgorithm._create_neural_network(weights, input_size, hidden_size, output_size)
    
    # Verifica se a rede neural foi criada corretamente
    assert isinstance(nn, NeuralNetwork)
    assert nn.weights_1.shape == (input_size, hidden_size)
    assert nn.weights_2.shape == (hidden_size, output_size)
    assert nn.bias_1.shape == (1, hidden_size)
    assert nn.bias_2.shape == (1, output_size)
