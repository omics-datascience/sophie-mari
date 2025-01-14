import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from deap import base, creator, tools, algorithms
import random

# Define fitness function for GA
def fitness_function(individual, X, y):
    """
    Fitness function for Genetic Algorithm to evaluate how well the selected features predict IC50 values.

    Parameters:
        individual (list): Binary list where 1 represents selecting the feature and 0 represents ignoring it.
        X (pd.DataFrame): Gene expression data matrix.
        y (list): IC50 values vector.

    Returns:
        tuple: (mean_squared_error, )
    """
    # Select features based on the binary representation in 'individual'
    selected_features = [i for i, bit in enumerate(individual) if bit == 1]
    
    if len(selected_features) == 0:
        return float('inf'),  # Penalize if no features are selected

    X_selected = X.iloc[:, selected_features]
    
    # Split the data into training and testing sets (90% train, 10% test)
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.1, random_state=42)
    
    # Train a KNN model on the training set
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train, y_train)
    
    # Predict and evaluate using Mean Squared Error
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return mse,  # Return a tuple (DEAP requires this format)

# Set up the GA
def run_ga(X, y, num_generations=20, population_size=50, mutation_prob=0.2, crossover_prob=0.7):
    """
    Run Genetic Algorithm to perform feature selection.

    Parameters:
        X (pd.DataFrame): Gene expression data matrix.
        y (list): IC50 values vector.
        num_generations (int): Number of generations to evolve the population.
        population_size (int): Number of individuals in each generation.
        mutation_prob (float): Probability of mutation for each individual.
        crossover_prob (float): Probability of crossover between two individuals.

    Returns:
        list: The best individual (list of selected features).
        float: The fitness score (MSE of the selected features).
    """
    # Define the GA environment
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize MSE
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Initialize the population with random binary individuals
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=mutation_prob)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", fitness_function, X=X, y=y)
    
    # Initialize the population
    population = toolbox.population(n=population_size)

    # Run the Genetic Algorithm
    for gen in range(num_generations):
        # Evaluate all individuals
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Select the next generation
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation
        for mutant in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the fitness of the offspring
        invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_individuals))
        for ind, fit in zip(invalid_individuals, fitnesses):
            ind.fitness.values = fit

        # Replace the old population with the new one
        population[:] = offspring

    # Get the best individual and its fitness score
    best_individual = tools.selBest(population, 1)[0]
    best_fitness = best_individual.fitness.values[0]
    
    return best_individual, best_fitness

