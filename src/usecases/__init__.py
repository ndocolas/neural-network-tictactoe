"""Módulo de casos de uso do jogo da velha com IA."""

from .genetic_algorithm import GeneticAlgorithm
from .score_evaluator import ScoreEvaluator

# Função de conveniência para treinar a rede neural
def train_neural_network(generations=50, population_size=50, games_per_generation=20, callback=None):
    """
    Treina uma rede neural usando algoritmo genético.
    
    Args:
        generations: Número de gerações para treinamento
        population_size: Tamanho da população
        games_per_generation: Número de jogos por avaliação
        callback: Função de callback para acompanhar o progresso
        
    Returns:
        Melhor cromossomo encontrado
    """
    ga = GeneticAlgorithm(
        population_size=population_size,
        generations=generations,
        n_games=games_per_generation
    )
    
    # Configura o callback se fornecido
    if callback:
        def wrapped_callback(generation, best_fitness):
            callback(generation, best_fitness)
            return best_fitness
        
        # Sobrescreve o método de evolução para incluir o callback
        original_evolve = ga.evolve
        
        def evolve_with_callback(*args, **kwargs):
            pop = ga._init_pop()
            best_chrom = None
            best_fit = -float("inf")

            for g in range(1, ga.generations + 1):
                # Avaliação
                for c in pop:
                    c.score = ga.evaluator.evaluate(c.weights_vector)
                
                # Ordena por score
                pop.sort(key=lambda c: c.score, reverse=True)
                
                # Atualiza melhor global
                if pop[0].score > best_fit:
                    best_fit = pop[0].score
                    best_chrom = pop[0]
                
                # Chama o callback
                wrapped_callback(g, best_fit)
                
                # Reprodução
                best = pop[0]
                next_pop = [best]  # elitismo
                
                while len(next_pop) < ga.pop_size:
                    p1 = ga._select_tournament(pop)
                    p2 = ga._select_tournament(pop)
                    while p2 is p1:
                        p2 = ga._select_tournament(pop)
                    
                    child = ga._crossover(p1, p2)
                    if g > round(ga.generations * 0.3):
                        ga._mutate(child)
                    next_pop.append(child)
                
                pop = next_pop
            
            return best_chrom.weights_vector if best_chrom else pop[0].weights_vector
        
        # Executa a evolução com callback
        return evolve_with_callback()
    
    # Executa a evolução sem callback
    return ga.evolve(verbose=True)
