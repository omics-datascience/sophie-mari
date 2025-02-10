from itertools import product
from typing import cast, Union, Final

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.neighbors import KNeighborsRegressor

# Parameters
D = 30  # Number of genes to select
K = 3  # Number of neighbors
NUM_RUNS = 100  # Monte Carlo iterations

# Result of GA metaheuristic = Tuple of the best features, the best model, and the best mean score
FSResult = tuple[list[str] | None, KNeighborsRegressor | None, float | None]

# Debug flag
DEBUG: Final[bool] = False

# MSE needs to be minimized
MORE_IS_BETTER = False

x_data = pd.read_csv('./gene_expression_and_ic50.csv', index_col=0).T.dropna()


def __compute_cross_validation(classifier: KNeighborsRegressor, subset: pd.DataFrame, y: np.ndarray) -> float:
    """
    Performs LOOCV using KNN and returns the mean squared error.
    """
    loo = LeaveOneOut()
    mse_list = []

    for train_index, test_index in loo.split(subset):
        x_train_fold, x_test_fold = subset.iloc[train_index], subset.iloc[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        cloned = clone(classifier)
        cloned = cast(KNeighborsRegressor, cloned)
        cloned.fit(x_train_fold.values, y_train_fold)
        try:
            y_predicted = cloned.predict(x_test_fold.values)
            score = mean_squared_error(y_test_fold, y_predicted)
        except Exception as ex:
            print(f"Error during CrossValidation: {ex}. Setting score to 0.0")
            score = 0.0
        
        mse_list.append(score)

    return np.mean(mse_list)


def __get_subset_of_features(molecules_df: pd.DataFrame, combination: np.ndarray) -> pd.DataFrame:
    """
    Gets a specific subset of features (genes) from a Pandas DataFrame.
    """
    if np.sum(combination) != D:  # Ensure exactly D genes are selected
        return pd.DataFrame()

    subset: pd.DataFrame = molecules_df.iloc[:, combination.astype(bool)].dropna()
    return subset


def genetic_algorithms(
        classifier: KNeighborsRegressor,
        molecules_df: pd.DataFrame,
        population_size: int,
        mutation_rate: float,
        n_iterations: int,
        y: np.ndarray,
        more_is_better: bool
) -> FSResult:
    n_genes = molecules_df.shape[1]
    population = np.random.randint(2, size=(population_size, n_genes))
    
    for _ in range(n_iterations):
        fitness_scores = np.array([
            __compute_cross_validation(classifier, __get_subset_of_features(molecules_df, solution), y)
            for solution in population
        ])
        
        if np.isnan(fitness_scores).any():
            raise ValueError("NaN values found in fitness scores. Check data preprocessing.")
        
        probabilities = fitness_scores / fitness_scores.sum()
        parents = population[np.random.choice(population_size, size=population_size, p=probabilities, replace=True)]
        
        crossover_point = np.random.randint(1, n_genes)
        offspring = np.zeros_like(population)
        for i in range(population_size // 2):
            parent1, parent2 = parents[i], parents[population_size - i - 1]
            offspring[i] = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            offspring[population_size - i - 1] = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        
        mask = np.random.rand(population_size, n_genes) < mutation_rate
        offspring[mask] = 1 - offspring[mask]
        population = offspring
    
    best_idx = np.argmax(fitness_scores) if more_is_better else np.argmin(fitness_scores)
    best_features_mask = population[best_idx].astype(bool)
    best_features_str = molecules_df.columns[best_features_mask].tolist()
    best_model = classifier.fit(molecules_df.loc[:, best_features_str], y)
    best_mean_score = fitness_scores[best_idx]
    
    return best_features_str, best_model, best_mean_score


if __name__ == '__main__':
    print('Reading data')
    x_data = pd.read_csv('./gene_expression_and_ic50.csv', index_col=0).T.dropna()
    y_data = x_data.iloc[-1].values
    x_data = x_data.iloc[:-1]

    print('Running Genetic Algorithms')
    result = genetic_algorithms(
        classifier=KNeighborsRegressor(n_neighbors=K, metric='euclidean'),
        molecules_df=x_data,
        population_size=50,
        mutation_rate=0.1,
        n_iterations=10,
        y=y_data,
        more_is_better=MORE_IS_BETTER
    )
    print(f'Best features: {len(result[0])} | Best MSE: {result[2]}')