from typing import cast, Union, Final

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

# Result of GA metaheuristic = Tuple of the best features, the best model, and the best mean score
FSResult = tuple[list[str] | None, KNeighborsRegressor | None, float | None]

# Debug flag
DEBUG: Final[bool] = True

# MSE need to be minimized
MORE_IS_BETTER = False


def __compute_cross_validation(classifier: KNeighborsRegressor, subset: pd.DataFrame, y: np.ndarray,
                               cross_validation_folds: int,
                               more_is_better: bool) -> tuple[float, KNeighborsRegressor, float]:
    """
    Computes CrossValidation to get the Concordance Index (using StratifiedKFold to prevent "All samples are censored"
    error).
    @param classifier: Classifier to train.
    @param subset: Subset of features to be used in the model evaluated in the CrossValidation.
    @param y: Classes.
    @param cross_validation_folds: Number of folds in the CrossValidation process.
    @return: Average of the C-Index obtained in each CV fold, best model during CV and its fitness score.
    """
    # Create StratifiedKFold object.
    skf = KFold(n_splits=cross_validation_folds, shuffle=True)
    lst_score_stratified: list[float] = []
    estimators: list[KNeighborsRegressor] = []

    # Makes the rows columns
    subset = subset.transpose()

    for train_index, test_index in skf.split(subset, y):
        # Splits
        x_train_fold, x_test_fold = subset.iloc[train_index], subset.iloc[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        # Creates a cloned instance of the model to store in the list. This HAVE TO be done before fit() because
        # clone() method does not clone the fit_X_ attribute (needed to restore the model during statistical
        # validations)
        cloned = clone(classifier)
        cloned = cast(KNeighborsRegressor, cloned)

        # Train and stores fitness
        cloned.fit(x_train_fold.values, y_train_fold)
        try:
            y_predicted = cloned.predict(x_test_fold.values)
            score = mean_squared_error(y_test_fold, y_predicted)
        except Exception as ex:
            print(f"Error during CrossValidation: {ex}. Setting score to 0.0")

            # To prevent issues with RF training with data that don't have any comparable pair
            score = 0.0

        lst_score_stratified.append(score)

        # Stores trained model
        estimators.append(cloned)

    # Gets best fitness
    if more_is_better:
        best_model_idx = np.argmax(lst_score_stratified)
    else:
        best_model_idx = np.argmin(lst_score_stratified)
    best_model = estimators[best_model_idx]
    best_fitness_value = lst_score_stratified[best_model_idx]
    fitness_value_mean = cast(float, np.mean(lst_score_stratified))

    if DEBUG:
        print(f'Testing {subset.shape[1]} features. MSE: {fitness_value_mean}')

    return fitness_value_mean, best_model, best_fitness_value


def __get_subset_of_features(molecules_df: pd.DataFrame, combination: Union[list[str], np.ndarray]) -> pd.DataFrame:
    """
    Gets a specific subset of features from a Pandas DataFrame.
    @param molecules_df: Pandas DataFrame with all the features.
    @param combination: Combination of features to extract.
    @return: A Pandas DataFrame with only the combinations of features.
    """
    # Get subset of features
    if isinstance(combination, np.ndarray):
        # NOTE: all the elements in combination must be booleans. Otherwise, Pandas returns all the rows
        if not np.issubdtype(combination.dtype, np.bool_):
            combination = combination.astype(bool)

        # In this case it's a Numpy array with int indexes (used in metaheuristics)
        subset: pd.DataFrame = molecules_df.iloc[combination]
    else:
        # In this case it's a list of columns names (used in Blind Search)
        molecules_to_extract = np.intersect1d(molecules_df.index.tolist(), combination)
        subset: pd.DataFrame = molecules_df.loc[molecules_to_extract]

    # Discards NaN values
    subset = subset[~pd.isnull(subset)]

    return subset


def genetic_algorithms(
        classifier: KNeighborsRegressor,
        molecules_df: pd.DataFrame,
        population_size: int,
        mutation_rate: float,
        n_iterations: int,
        y: np.ndarray,
        cross_validation_folds: int,
        more_is_better: bool
) -> FSResult:
    # Initialize population randomly
    n_molecules = molecules_df.shape[0]
    population = np.random.randint(2, size=(population_size, n_molecules))

    fitness_scores = np.empty((population_size, 2))

    for _iteration in range(n_iterations):
        # Calculate fitness scores for each solution
        fitness_scores = np.array([
            __compute_cross_validation(
                classifier,
                subset=__get_subset_of_features(molecules_df, combination=solution),
                y=y,
                cross_validation_folds=cross_validation_folds,
                more_is_better=more_is_better
            )
            for solution in population
        ])

        # Gets scores and casts the type to float to prevent errors due to 'safe' option
        scores = fitness_scores[:, 0].astype(float)

        # Select parents based on fitness scores
        parents = population[
            np.random.choice(population_size, size=population_size, p=scores / scores.sum())
        ]

        # Crossover (single-point crossover)
        crossover_point = np.random.randint(1, n_molecules)
        offspring = np.zeros_like(population)
        for i in range(population_size // 2):
            parent1, parent2 = parents[i], parents[population_size - i - 1]
            offspring[i] = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            offspring[population_size - i - 1] = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

        # Mutation
        mask = np.random.rand(population_size, n_molecules) < mutation_rate
        offspring[mask] = 1 - offspring[mask]

        population = offspring

    # Get the best solution
    best_idx = np.argmax(fitness_scores[:, 0]) if more_is_better else np.argmin(fitness_scores[:, 0])
    best_features = population[best_idx]
    best_features = best_features.astype(bool)  # Pandas needs a boolean array to select the rows
    best_features_str: list[str] = molecules_df.iloc[best_features].index.tolist()

    best_model = cast(KNeighborsRegressor, fitness_scores[best_idx][1])
    best_mean_score = cast(float, fitness_scores[best_idx][0])
    return best_features_str, best_model, best_mean_score


if __name__ == '__main__':
    # Reads data from CSV file (last row is the target)
    x_data = pd.read_csv('./gene_expression_and_ic50.csv', index_col=0)
    y_data = x_data.iloc[-1].values
    x_data = x_data.iloc[:-1]

    result = genetic_algorithms(
        classifier=KNeighborsRegressor(),
        molecules_df=x_data,
        population_size=10,
        mutation_rate=0.1,
        n_iterations=10,
        y=y_data,
        cross_validation_folds=5,
        more_is_better=MORE_IS_BETTER
    )

    if DEBUG:
        original_mse_mean = __compute_cross_validation(
            classifier=KNeighborsRegressor(),
            subset=x_data,
            y=y_data,
            cross_validation_folds=5,
            more_is_better=MORE_IS_BETTER
        )[0]
        print(f'Started with {x_data.shape[0]} features. Original MSE: {original_mse_mean}')
        print(f'Best features: {len(result[0])} | Best MSE: {result[2]} | Best model instance: {result[1]}')
