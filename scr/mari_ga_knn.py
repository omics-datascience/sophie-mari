import numpy as np
from sklearn.neighbors import NearestNeighbors
import random

print('test1')

def monte_carlo_cross_validation(X, y, num_genes=30, k=3, num_partitions=100):
    """
    Perform Monte Carlo Cross-Validation combined with genetic algorithm-based feature selection.

    Parameters:
    X (ndarray): The input matrix of gene expression values (genes x samples).
    y (ndarray): The vector of IC50 values for each sample.
    num_genes (int): Number of genes to select for predictive modeling.
    k (int): Number of neighbors for k-nearest neighbors algorithm.
    num_partitions (int): Number of Monte Carlo partitions.

    Returns:
    tuple: Final predicted IC50 values for training and testing samples.
    """
    n_genes, n_samples = X.shape  # Now X.shape is (genes, samples)
    
    # Store predictions for training and testing samples
    train_predictions = np.zeros(n_samples)
    test_predictions = np.zeros(n_samples)
    
    # Count appearances in training and testing sets
    train_counts = np.zeros(n_samples)
    test_counts = np.zeros(n_samples)

    # Monte Carlo Cross-Validation
    for partition in range(num_partitions):
        # Randomly split data
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        train_size = int(0.9 * n_samples)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        X_train, y_train = X[:, train_indices], y[train_indices]
        X_test, y_test = X[:, test_indices], y[test_indices]

        # Feature selection using Genetic Algorithm
        selected_genes = genetic_algorithm(X_train, y_train, num_genes, k)
        
        # Use selected genes
        X_train_selected = X_train[selected_genes, :]  # Select rows (genes) for training
        X_test_selected = X_test[selected_genes, :]    # Select rows (genes) for testing

        # KNN Prediction for Test Set
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X_train_selected.T)  # Transpose X for fitting (samples x genes)
        distances, neighbors = nbrs.kneighbors(X_test_selected.T)
        for i, sample_index in enumerate(test_indices):
            neighbor_indices = neighbors[i]
            test_predictions[sample_index] += np.mean(y_train[neighbor_indices])
            test_counts[sample_index] += 1

        # KNN Prediction for Training Set (LOOCV)
        for i, sample_index in enumerate(train_indices):
            loo_train_indices = [index for index in train_indices if index != sample_index]  # Manually exclude sample_index
            loo_X_train = X_train_selected[:, loo_train_indices]  # Select all genes, but exclude current sample
            loo_y_train = y_train[loo_train_indices]

            nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(loo_X_train.T)  # Fit on the transposed training data
            distances, neighbors = nbrs.kneighbors(X_train_selected[:, i].reshape(1, -1).T)  # Reshape for fitting, transpose for proper alignment
            train_predictions[sample_index] += np.mean(loo_y_train[neighbors[0]])
            train_counts[sample_index] += 1

    # Compute final predictions by averaging over partitions
    train_final = np.where(train_counts > 0, train_predictions / train_counts, 0)
    test_final = np.where(test_counts > 0, test_predictions / test_counts, 0)

    return train_final, test_final

def genetic_algorithm(X, y, num_genes, k, population_size=50, generations=100, mutation_rate=0.01):
    """
    Perform genetic algorithm-based feature selection to identify predictive gene subsets.

    Parameters:
    X (ndarray): The input matrix of gene expression values (genes x samples).
    y (ndarray): The vector of IC50 values for each sample.
    num_genes (int): Number of genes to select.
    k (int): Number of neighbors for k-nearest neighbors algorithm.
    population_size (int): Size of the population in the genetic algorithm.
    generations (int): Number of generations to run the genetic algorithm.
    mutation_rate (float): Probability of mutation for each gene in a chromosome.

    Returns:
    ndarray: Indices of the selected genes.
    """
    n_genes, n_samples = X.shape  # X is (genes, samples)

    # Fitness function: Squared-error loss
    def fitness(individual):
        selected_genes = np.where(individual == 1)[0]
        if len(selected_genes) != num_genes:
            return float('inf')  # Penalize invalid subsets

        X_selected = X[selected_genes, :]
        errors = []

        for i in range(n_samples):
            loo_train_X = np.delete(X_selected, i, axis=1)  # Remove column (sample) i
            loo_train_y = np.delete(y, i)
            test_sample = X_selected[:, i]

            nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(loo_train_X.T)
            distances, neighbors = nbrs.kneighbors(test_sample.reshape(-1, 1).T)
            prediction = np.mean(loo_train_y[neighbors[0]])
            errors.append((prediction - y[i])**2)

        return np.mean(errors)

    # Initialize population
    population = [np.random.choice([0, 1], size=n_genes, p=[1 - num_genes / n_genes, num_genes / n_genes]) for _ in range(population_size)]

    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = np.array([fitness(ind) for ind in population])

        # Select individuals (tournament selection)
        selected_indices = np.argsort(fitness_scores)[:population_size // 2]
        selected_population = [population[i] for i in selected_indices]

        # Crossover
        children = []
        while len(children) < population_size - len(selected_population):
            parent1, parent2 = random.sample(selected_population, 2)
            crossover_point = random.randint(1, n_genes - 1)
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            children.append(child)

        # Mutation
        for child in children:
            if random.random() < mutation_rate:
                mutation_index = random.randint(0, n_genes - 1)
                child[mutation_index] = 1 - child[mutation_index]

        # Update population
        population = selected_population + children

    # Return best individual
    best_individual = population[np.argmin([fitness(ind) for ind in population])]
    return np.where(best_individual == 1)[0]


np.random.seed(42)

# Mock data
num_samples = 100
num_genes = 100
X_mock = np.random.rand(num_genes, num_samples)  # Random gene expression values, genes x samples
y_mock = np.random.rand(num_samples)  # Random IC50 values

train_preds, test_preds = monte_carlo_cross_validation(X_mock, y_mock, num_genes=30, k=3, num_partitions=10)
print("Train Predictions:", train_preds[:10])  # Print first 10 predictions
print("Test Predictions:", test_preds[:10])
