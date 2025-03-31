import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from typing import Union, cast
from collections import Counter
from itertools import product
from tqdm import tqdm


# Set the seed
np.random.seed(42)

# Result of GA metaheuristic = Tuple of the best features, the best model, and the best mean score
FSResult = tuple[list[str] | None, KNeighborsRegressor | None, float | None]

# Debug flag
DEBUG = False

# MSE needs to be minimized
MORE_IS_BETTER = False

def feature_selection(population, max_features=30, num_features=20242):
    """
    Ensures that the feature selection process is limited to `max_features` features per individual.
    population: list of individuals (solutions).
    max_features: max number of features allowed.
    num_features: total number of features (length of the feature vector).
    """
    for i, individual in enumerate(population):
        # Ensure that each individual has at most `max_features` selected features
        selected_features_indices = np.random.choice(num_features, size=max_features, replace=False)
        
        # Reset individual to a zeroed array of shape (num_features,)
        new_individual = np.zeros(num_features, dtype=int)
        
        # Set the selected features to 1
        new_individual[selected_features_indices] = 1
        
        # Assign back to the population
        population[i] = new_individual
        
    return population


def __compute_cross_validation(classifier: KNeighborsRegressor, subset: pd.DataFrame, y: np.ndarray,
                               more_is_better: bool) -> tuple[float, KNeighborsRegressor, float]:
    """
    Computes CrossValidation using Leave-One-Out (LOO).
    """
    loo = LeaveOneOut()
    lst_score_stratified: list[float] = []
    estimators: list[KNeighborsRegressor] = []

    for train_index, test_index in loo.split(subset, y):
        # Splitting data
        x_train_fold, x_test_fold = subset.iloc[train_index], subset.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        # Clone model
        cloned = clone(classifier)
        cloned = cast(KNeighborsRegressor, cloned)

        # Train and evaluate
        cloned.fit(x_train_fold.values, y_train_fold)
        try:
            y_predicted = cloned.predict(x_test_fold.values)
            score = mean_squared_error(y_test_fold, y_predicted)
        except Exception as ex:
            print(f"Error during CrossValidation: {ex}. Setting score to 0.0")
            score = 0.0

        lst_score_stratified.append(score)
        estimators.append(cloned)

    # Select best model
    best_model_idx = np.argmax(lst_score_stratified) if more_is_better else np.argmin(lst_score_stratified)
    best_model = estimators[best_model_idx]
    best_fitness_value = lst_score_stratified[best_model_idx]
    fitness_value_mean = float(np.mean(lst_score_stratified))

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
        subset: pd.DataFrame = molecules_df.iloc[:, combination]

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
        more_is_better: bool,
        max_features: int = 30  # Max number of features allowed per individual
) -> FSResult:
    
    # Set the maximum number of features to select
    n_molecules = molecules_df.shape[1]  # Total number of features in molecules_df

    # Initialize population randomly with a boolean array of size n_features
    population = np.zeros((population_size, n_molecules), dtype=int)

    for i in range(population_size):
        # Randomly select at most 30 features to be True (selected)
        selected_features = np.random.choice(n_molecules, size=max_features, replace=False)
        population[i, selected_features] = 1

    # Perform feature selection on initial population
    population = feature_selection(population, max_features=30, num_features=20242)

    for _iteration in range(n_iterations):
        # Calculate fitness scores for each solution
        fitness_scores = np.array([
            __compute_cross_validation(
                classifier,
                subset=__get_subset_of_features(molecules_df, combination=solution),
                y=y,
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

        # Perform feature selection after mutation and crossover
        population = feature_selection(offspring, max_features)

    # Get the best solution
    best_idx = np.argmax(fitness_scores[:, 0]) if more_is_better else np.argmin(fitness_scores[:, 0])
    best_features = population[best_idx]
    best_features = best_features.astype(bool)  # Pandas needs a boolean array to select the rows
    selected_feature_indices = np.where(best_features)[0]  # Get indices of selected features
    best_features_str: list[str] = molecules_df.columns[selected_feature_indices].tolist()

    best_model = cast(KNeighborsRegressor, fitness_scores[best_idx][1])
    best_mean_score = cast(float, fitness_scores[best_idx][0])
    return best_features_str, best_model, best_mean_score


def monte_carlo_sample_selection(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, num_splits: int = 100):
    """
    Performs Monte Carlo Cross-Validation to select different training samples for the Genetic Algorithm.
    
    Parameters:
    - X: Feature matrix (DataFrame)
    - y: Target labels (Series)
    - test_size: Fraction of the dataset to be used for testing
    - num_splits: Number of Monte Carlo iterations
    
    Returns:
    - List of (X_train, X_test, y_train, y_test) splits
    """
    mc_splits = []
    
    for _ in tqdm(range(num_splits), desc="Generating MCCV Splits", leave=False):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size
        )

        mc_splits.append((X_train, X_test, y_train, y_test))
    
    return mc_splits


# Constants
MORE_IS_BETTER = True  # Set this flag according to your definition
NUM_SPLITS = 100  # Set number of random splits to 100
FEATURES_SELECTED = 30  # Number of features to select per split

# Lists to store all predictions for averaging later
train_predictions = []
test_predictions = []

# List to store selected features across splits
all_selected_features = []

# Best performance tracking
best_mse = float('inf') if not MORE_IS_BETTER else float('-inf')
best_model = None
best_features = None

if __name__ == '__main__':
    # Gets the preprocessed data
    print('Loading data')
    x_data_og = pd.read_csv('./gene_expression_drug1.csv', index_col=0)  # (20242, 41)
    y_data = pd.read_csv('./IC50_drug1.csv', index_col=0)  # (41, 1)
    y_data = y_data.squeeze()  # Convert DataFrame (41,1) to Series (41,)

    # Ensure x_data remains (41, 20242) for proper training
    x_data = x_data_og.T  # Transpose so rows become samples and columns become features

    print("X shape:", x_data.shape)  # Should be (20242, 41)
    print("y shape:", y_data.shape)  # Should be (41, )


    # Reports original MSE with all the features
    print('Running fitness function with all the features')
    original_mse_mean = __compute_cross_validation(
        classifier=KNeighborsRegressor(),
        subset=x_data,
        y=y_data,
        more_is_better=MORE_IS_BETTER
    )[0]
    print(f'Started with {x_data.shape[0]} features. Original MSE: {original_mse_mean}')

    # Monte Carlo Cross-Validation for different train-test splits
    print('\nRunning Monte Carlo Cross-Validation')

    # Define a list to store the selected features from each MCCV split
    all_selected_features = []
    mc_splits = monte_carlo_sample_selection(X=x_data, y=y_data, test_size=0.2, num_splits=NUM_SPLITS)

    # MCCV loop
    for i, (X_train, X_test, y_train, y_test) in enumerate(tqdm(mc_splits, desc="MCCV Progress")):
        print(f"Running MCCV split {i+1}/{len(mc_splits)}")
        
        # Run the genetic algorithm on the training set
        print("Running Genetic Algorithm for feature selection")
        selected_features, trained_knn_model, best_score = genetic_algorithms(
            classifier=KNeighborsRegressor(),
            molecules_df=X_train,  # Train set for feature selection
            population_size=10,
            mutation_rate=0.1,
            n_iterations=10,
            y=y_train,  # Target variable from train set
            more_is_better=MORE_IS_BETTER
        )

        print(f"Split {i+1}: Selected {len(selected_features)} features | Best MSE = {best_score}")
        print(f"Selected features (indices): {selected_features}")

        # If selected_features is empty, print warning
        if not selected_features:
            print(f"Warning: No features selected in MCCV split {i+1}")
            continue  # Skip this MCCV split if no features were selected

        # Store selected features from this split
        all_selected_features.append(selected_features)

        # Track best model & feature selection process
        if best_score > best_mse:
            print(f"New best model found! MSE improved from {best_mse} → {best_score}")
            best_mse = best_score
            best_model = trained_knn_model
            best_features = selected_features

        print("X_train:", X_train)
        print("X_train shape:", X_train.shape)
        print("Selected features:", selected_features)

        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        print(f"X_train_selected shape: {X_train_selected.shape}")
        
        # Train k-NN (k=3) using only the selected features
        knn = KNeighborsRegressor(n_neighbors=3)
        knn.fit(X_train_selected, y_train)

        # Predict IC50 for test set using k=3 nearest neighbors
        y_pred_test = knn.predict(X_test_selected)
        print(f"Split {i+1}: Test set predictions computed.")

    # Print final best model details
    print(f"\nBest MSE from MCCV: {best_mse}")
    print(f"Best model: {best_model}")

    if best_features is None:
        print("Error: best_features was never assigned.")

    print(f"Best features selected: {len(best_features)}")

    # Aggregate ranked genes by frequency of selection
    all_genes = [gene for gene_list in all_selected_features for gene in gene_list]
    gene_counts = Counter(all_genes)

    # Convert to DataFrame and sort by selection frequency
    gene_frequency_df = pd.DataFrame.from_dict(gene_counts, orient='index', columns=['Frequency'])

    # Map numerical indices to gene names from the saved matrix
    gene_frequency_df.index = x_data_og.index[list(gene_frequency_df.index)]
    gene_frequency_df.index.name = 'Gene'

    # Sort genes by selection frequency
    gene_frequency_df = gene_frequency_df.sort_values(by='Frequency', ascending=False)

    # Display top-ranked genes
    print("\nTop selected genes:")
    print(gene_frequency_df)
