import requests
import random
import json
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# API Configuration
API_URL = "https://bioapi.multiomix.org/expression-of-genes"
HEADERS = {"Content-Type": "application/json"}
TISSUE = "Skin"  # Specify the tissue of interest

# GA Parameters
NUM_GENES = 30     # Number of genes to select (subset size)
POP_SIZE = 50      # Population size
N_GENERATIONS = 20 # Number of generations
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.2

# Training and Test IC50 Data (Simulated for Example)
# Replace these with actual IC50 data relevant to the study
# 90:10 ratio
y_train = [1.5, 2.3, 0.9, 1.8]  # Example training IC50 values
y_test = [1.7, 2.1, 1.0]        # Example testing IC50 values

# Helper: Fetch gene expression data using the API
def fetch_gene_expression(gene_ids, tissue=TISSUE):
    """
    Fetch gene expression data from the API for the given genes and tissue.
    """
    body = {
        "tissue": tissue,
        "gene_ids": gene_ids,
        "type": "json"
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(body))
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    
def knn_predict_with_self_exclusion(X_train, y_train, X_test=None, k=3, predict_train=False):
    """
    Implements KNN prediction for IC50 values with optional self-exclusion for training data.
    
    Args:
        X_train (np.ndarray): Training gene expression data.
        y_train (np.ndarray): Observed IC50 values for the training set.
        X_test (np.ndarray or None): Testing gene expression data. If None, predict for training data with self-exclusion.
        k (int): Number of nearest neighbors.
        predict_train (bool): Whether to predict for training samples with self-exclusion.
        
    Returns:
        y_pred (list): Predicted IC50 values.
    """
    y_pred = []

    # Determine if we are predicting for training data with self-exclusion
    if predict_train:
        X_test = X_train  # Use training data as the test set

    for test_index, test_sample in enumerate(X_test):
        # Compute distances from the test sample to all training samples
        distances = np.linalg.norm(X_train - test_sample, axis=1)
        
        # Exclude self if predicting for training data
        if predict_train:
            distances[test_index] = np.inf  # Set distance to self as infinity to exclude it
        
        # Find indices of the k closest training samples
        nearest_neighbor_indices = np.argsort(distances)[:k]
        
        # Average the IC50 values of the k nearest neighbors
        nearest_neighbor_ic50 = [y_train[i] for i in nearest_neighbor_indices]
        predicted_ic50 = np.mean(nearest_neighbor_ic50)
        y_pred.append(predicted_ic50)

    return y_pred

# GA Evaluation Function
def evaluate_individual(individual):
    """
    Evaluate the fitness of an individual by fetching gene expression data
    and using it to predict IC50 values.
    """
    # Select the genes (1 = selected, 0 = not selected)
    selected_genes = [gene_list[i] for i in range(len(individual)) if individual[i] == 1]
    
    if not selected_genes:  # No genes selected
        return 100.0,  # Penalize heavily for no selection
    
    # Fetch expression data for selected genes
    expression_data = fetch_gene_expression(selected_genes)
    if not expression_data:
        return 100.0,  # Penalize heavily for API failure
    
    # Convert expression data into a feature matrix (X_train, X_test)

    total_samples = len(expression_data)
    train_samples = len(y_train)
    test_samples = len(y_test)
    
    if total_samples < train_samples + test_samples:
        print("Error: Not enough samples returned from the API.")
        return 100.0,  # Penalize heavily for insufficient samples
    
    # Split the expression data into training and testing sets
    X_train = [[sample[gene] for gene in selected_genes] for sample in expression_data[:train_samples]]
    X_test = [[sample[gene] for gene in selected_genes] for sample in expression_data[train_samples:train_samples + test_samples]]
    
    # Train and evaluate KNN model, using training data and self exclusion
    y_pred = knn_predict_with_self_exclusion(X_train=np.array(X_train), 
                                               y_train=np.array(y_train), 
                                               k=3, 
                                               predict_train=True)


    
    # Calculate mean squared error as the fitness value
    mse = mean_squared_error(y_train, y_pred)
    return mse,

# Initialize GA
gene_list = ["BRCA1", "BRCA2", "TP53", "EGFR"]  # Replace with the full list of genes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(gene_list))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run the Genetic Algorithm
def run_genetic_algorithm():
    population = toolbox.population(n=POP_SIZE)
    for gen in range(N_GENERATIONS):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_RATE:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < MUTATION_RATE:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population with offspring
        population[:] = offspring
    
    # Return the best individual
    best_individual = tools.selBest(population, 1)[0]
    selected_genes = [gene_list[i] for i in range(len(best_individual)) if best_individual[i] == 1]
    print("Best gene subset:", selected_genes)
    print("Best fitness (MSE):", best_individual.fitness.values[0])

# Run GA
