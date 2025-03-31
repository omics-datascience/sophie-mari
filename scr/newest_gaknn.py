import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from typing import Tuple, List, Union, cast, Dict
from collections import Counter
from collections import defaultdict
from tqdm import tqdm
import warnings
from pathlib import Path

# Suppress sklearn warnings
warnings.filterwarnings('ignore')

# Set the seed for reproducibility
np.random.seed(42)

# Type aliases
FSResult = Tuple[List[str], KNeighborsRegressor, float]
FeatureMask = Union[List[str], np.ndarray]

# Configuration
DEBUG = False
MORE_IS_BETTER = False  # MSE needs to be minimized
NUM_SPLITS = 100        # Number of Monte Carlo splits
MAX_FEATURES = 30       # Max features per individual
POPULATION_SIZE = 10    # GA population size
MUTATION_RATE = 0.1     # GA mutation rate
N_ITERATIONS = 10       # GA iterations


def initialize_population(population_size: int, num_features: int, max_features: int) -> np.ndarray:
    """Initialize population with random feature subsets."""
    population = np.zeros((population_size, num_features), dtype=bool)
    for i in range(population_size):
        selected = np.random.choice(num_features, size=max_features, replace=False)
        population[i, selected] = True
    return population


def limit_features(population: np.ndarray, max_features: int) -> np.ndarray:
    """Ensure individuals don't exceed max_features by randomly removing excess features."""
    for i in range(len(population)):
        selected = np.where(population[i])[0]
        if len(selected) > max_features:
            to_remove = np.random.choice(selected, size=len(selected)-max_features, replace=False)
            population[i, to_remove] = False
    return population


def compute_cross_validation(
    model: KNeighborsRegressor, 
    X: pd.DataFrame, 
    y: pd.Series,
    more_is_better: bool = False
) -> Tuple[float, KNeighborsRegressor, float]:
    """Perform Leave-One-Out cross-validation and return metrics."""
    loo = LeaveOneOut()
    scores = []
    models = []

    for train_idx, test_idx in loo.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        cloned = clone(model)
        cloned.fit(X_train, y_train)
        
        try:
            y_pred = cloned.predict(X_test)
            score = mean_squared_error(y_test, y_pred)
        except Exception as e:
            if DEBUG:
                print(f"CV error: {e}")
            score = np.inf if not more_is_better else -np.inf

        scores.append(score)
        models.append(cloned)

    # Select best model
    best_idx = np.argmax(scores) if more_is_better else np.argmin(scores)
    best_model = models[best_idx]
    best_score = scores[best_idx]
    mean_score = np.mean(scores)

    if DEBUG:
        print(f'Features: {X.shape[1]}, Mean MSE: {mean_score:.4f}')

    return mean_score, best_model, best_score


def get_feature_subset(data: pd.DataFrame, mask: FeatureMask) -> pd.DataFrame:
    """Return subset of features based on selection mask."""
    if isinstance(mask, np.ndarray):
        if mask.dtype != bool:
            mask = mask.astype(bool)
        return data.iloc[:, mask]
    return data[list(mask)]


def genetic_algorithm(
    model: KNeighborsRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    population_size: int = POPULATION_SIZE,
    mutation_rate: float = MUTATION_RATE,
    n_iterations: int = N_ITERATIONS,
    max_features: int = MAX_FEATURES,
    more_is_better: bool = MORE_IS_BETTER
) -> FSResult:
    """Genetic algorithm for feature selection."""
    n_features = X.shape[1]
    population = initialize_population(population_size, n_features, max_features)
    best_solution = None
    best_score = np.inf if not more_is_better else -np.inf
    
    for iteration in tqdm(range(n_iterations), desc="GA Progress", disable=not DEBUG):
        # Evaluate population
        fitness = []
        for individual in population:
            X_subset = get_feature_subset(X, individual)
            score, current_model, _ = compute_cross_validation(model, X_subset, y, more_is_better)
            fitness.append((score, current_model, individual))
            
            # Track best solution
            if (more_is_better and score > best_score) or (not more_is_better and score < best_score):
                best_score = score
                best_solution = (individual.copy(), current_model, score)
        
        # Selection - tournament selection
        parents = []
        for _ in range(population_size):
            candidates = np.random.choice(len(fitness), size=3, replace=False)
            winner = max(candidates, key=lambda x: fitness[x][0]) if more_is_better else min(candidates, key=lambda x: fitness[x][0])
            parents.append(population[winner])
        
        # Crossover - uniform crossover
        offspring = np.zeros_like(population)
        for i in range(0, population_size, 2):
            if i+1 >= population_size:
                offspring[i] = parents[i]
                continue
                
            mask = np.random.rand(n_features) < 0.5
            offspring[i] = np.where(mask, parents[i], parents[i+1])
            offspring[i+1] = np.where(mask, parents[i+1], parents[i])
        
        # Mutation - bit flip
        mutation_mask = np.random.rand(population_size, n_features) < mutation_rate
        offspring = np.where(mutation_mask, ~offspring, offspring)
        
        # Apply feature limit
        population = limit_features(offspring, max_features)
    
    # Get best features
    if best_solution is None:
        raise RuntimeError("GA failed to find any solution")
    
    best_mask, best_model, best_score = best_solution
    feature_names = X.columns[best_mask].tolist()
    return feature_names, best_model, best_score


def monte_carlo_validation(
    X: pd.DataFrame, 
    y: pd.Series, 
    model: KNeighborsRegressor,
    n_splits: int = NUM_SPLITS,
    test_size: float = 0.2
) -> Tuple[List[List[str]], KNeighborsRegressor, float]:
    """Perform Monte Carlo Cross-Validation with feature selection."""
    all_selected = []
    best_model = None
    best_score = np.inf if not MORE_IS_BETTER else -np.inf
    
    for _ in tqdm(range(n_splits), desc="MCCV Progress"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        # Feature selection on training set
        features, model, score = genetic_algorithm(
            model=model,
            X=X_train,
            y=y_train,
            population_size=POPULATION_SIZE,
            mutation_rate=MUTATION_RATE,
            n_iterations=N_ITERATIONS,
            max_features=MAX_FEATURES,
            more_is_better=MORE_IS_BETTER
        )
        
        all_selected.append(features)
        
        # Track best model
        if (MORE_IS_BETTER and score > best_score) or (not MORE_IS_BETTER and score < best_score):
            best_score = score
            best_model = model
    
    return all_selected, best_model, best_score


def analyze_feature_frequency(feature_lists: List[List[str]], feature_names: pd.Index) -> pd.DataFrame:
    """Analyze how frequently each feature was selected."""
    counter = Counter([f for sublist in feature_lists for f in sublist])
    freq_df = pd.DataFrame.from_dict(counter, orient='index', columns=['Frequency'])
    freq_df.index.name = 'Feature'
    return freq_df.sort_values('Frequency', ascending=False)


# Define the exact paths you're using
GENE_EXPR_DIR = "/Users/marianajannotti/Documents/GitHub/sophie-mari/GENE_EXPR_DIR"
IC50_DIR = "/Users/marianajannotti/Documents/GitHub/sophie-mari/IC50_DIR"
RESULTS_DIR = "/Users/marianajannotti/Documents/GitHub/sophie-mari/results"

# Create results directory if it doesn't exist
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

def load_drug_data(drug_id: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Load data from your exact directory structure"""
    # Using pathlib for robust path handling
    gene_path = Path(GENE_EXPR_DIR) / f"{drug_id}_matrix.csv"
    ic50_path = Path(IC50_DIR) / f"{drug_id}_vector.csv"
    
    # Debugging output to verify paths
    print(f"Looking for gene file at: {gene_path}")
    print(f"Looking for IC50 file at: {ic50_path}")
    
    if not gene_path.exists():
        print(f"❌ Gene expression file not found at: {gene_path}")
        return None, None
    if not ic50_path.exists():
        print(f"❌ IC50 file not found at: {ic50_path}")
        return None, None
    
    try:
        X = pd.read_csv(gene_path, index_col=0).T  # Genes as columns
        y = pd.read_csv(ic50_path, header=None)
        
        # Diagnostic output with labels
        print(f"\n=== Checking file: {ic50_path} ===")
        print("First two lines of IC50 file:")
        with open(ic50_path) as f:  # Use the path string directly
            print("Line 1 (header?):", repr(f.readline()))  # repr() shows hidden characters
            print("Line 2 (1st value):", repr(f.readline()))
        
        # Dimension check
        print(f"\nShape check:")
        print(f"Gene matrix samples: {X.shape[0]}")
        print(f"IC50 values: {len(y)}")
        
        if len(y) == X.shape[0] + 1:
            print("\n⚠️ Warning: Off-by-one mismatch detected (likely header row)")
            print("Attempting automatic correction...")
            y = pd.read_csv(ic50_path, header=None, skiprows=1).squeeze()
            print(f"New IC50 count: {len(y)} (should now match {X.shape[0]})")
            
        return X, y
    except Exception as e:
        print(f"❌ Error loading drug {drug_id}: {str(e)}")
        return None, None


def process_drug(drug_id: int, model: KNeighborsRegressor) -> Dict:
    """Process a single drug through the full pipeline."""
    X, y = load_drug_data(drug_id)
    if X is None or y is None:
        return None
    
    print(f"\nProcessing drug {drug_id} with {X.shape[1]} genes and {len(y)} samples")
    
    # Store all results for this drug
    drug_results = {
        'selected_features': [],
        'predictions': [],
        'actual_values': y.values,
        'models': []
    }
    
    # Run MCCV
    for split in tqdm(range(NUM_SPLITS), desc=f"Drug {drug_id}"):
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=split
        )
        
        # Feature selection
        features, knn_model, _ = genetic_algorithm(
            model=model,
            X=X_train,
            y=y_train,
            population_size=POPULATION_SIZE,
            mutation_rate=MUTATION_RATE,
            n_iterations=N_ITERATIONS,
            max_features=MAX_FEATURES,
            more_is_better=MORE_IS_BETTER
        )
        
        # Predict on test set
        X_test_selected = X_test[features]
        y_pred = knn_model.predict(X_test_selected)
        
        # Store results
        drug_results['selected_features'].append(features)
        drug_results['predictions'].append(y_pred)
        drug_results['models'].append(knn_model)
    
    return drug_results


def save_drug_results(drug_id: int, results: Dict):
    """Save all results for a drug in an organized structure."""
    drug_dir = os.path.join(RESULTS_DIR, str(drug_id))
    os.makedirs(drug_dir, exist_ok=True)
    
    # 1. Save selected features
    features_df = pd.DataFrame(results['selected_features']).T
    features_df.to_csv(os.path.join(drug_dir, "selected_features.csv"))
    
    # 2. Save predictions (each split as a column)
    preds_df = pd.DataFrame(results['predictions']).T
    preds_df['actual'] = results['actual_values']
    preds_df.to_csv(os.path.join(drug_dir, "predictions.csv"))
    
    # 3. Save aggregated predictions (mean across all splits)
    mean_preds = np.mean(results['predictions'], axis=0)
    final_df = pd.DataFrame({
        'actual': results['actual_values'],
        'predicted': mean_preds
    })
    final_df.to_csv(os.path.join(drug_dir, "final_predictions.csv"))
    
    # 4. Save feature frequencies
    feature_counts = Counter([
        f for sublist in results['selected_features'] for f in sublist
    ])
    freq_df = pd.DataFrame.from_dict(
        feature_counts, orient='index', columns=['count']
    ).sort_values('count', ascending=False)
    freq_df.to_csv(os.path.join(drug_dir, "feature_frequencies.csv"))
    
    print(f"Saved results for drug {drug_id} in {drug_dir}")


def main():
    # Initialize model
    knn = KNeighborsRegressor(n_neighbors=3)
    
    # Load drug IDs
    drug_ids = [
        1, 3, 5, 6, 9, 11, 17, 29, 30, 32, 34, 35, 37, 38, 41, 45, 51, 52, 
        53, 54, 55, 56, 59, 60, 62, 63, 64, 71, 83, 86, 87, 88, 89, 91, 94, 
        104, 106, 110, 111, 119, 127, 133, 134, 135, 136, 140, 147, 150, 151, 
        152, 153, 154, 155, 156, 157, 158, 159, 163, 164, 165, 166, 167, 170, 
        171, 172, 173, 175, 176, 177, 178, 179, 180, 182, 184, 185, 186, 190, 
        192, 193, 194, 196, 197, 199, 200, 201, 202, 203, 204, 205, 206, 207, 
        208, 211, 219, 221, 222, 223, 224, 225, 226, 228, 229, 230, 231, 235, 
        236, 238, 245, 249, 252, 253, 254, 255, 256, 257, 258, 260, 261, 262, 
        263, 264, 265, 266, 268, 269, 271, 272, 273, 274, 275, 276, 277, 279, 
        281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 
        295, 298, 299, 300, 301, 302, 303, 304, 305, 306, 308, 309, 310, 312, 
        317, 326, 328, 329, 330, 331, 332, 333, 341, 342, 344, 345, 346, 356, 
        362, 363, 366, 371, 372, 374, 375, 376, 380, 381, 382, 406, 407, 408, 
        409, 410, 412, 415, 416, 427, 428, 431, 432, 435, 436, 437, 438, 439, 
        442, 446, 447, 449, 461, 474, 476, 477, 478, 546, 552, 562, 563, 573, 
        574, 576, 1001, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 
        1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 
        1025, 1026, 1028, 1029, 1030, 1031, 1032, 1033, 1036, 1037, 1038, 1039, 
        1042, 1043, 1046, 1047, 1048, 1049, 1050, 1052, 1053, 1054, 1057, 1058, 
        1059, 1060, 1061, 1062, 1066, 1067, 1069, 1072, 1091, 1114, 1129, 1133, 
        1142, 1143, 1149, 1158, 1161, 1164, 1166, 1170, 1175, 1192, 1194, 1199, 
        1203, 1218, 1219, 1230, 1236, 1239, 1241, 1242, 1243, 1248, 1259, 1261, 
        1262, 1263, 1264, 1266, 1268, 1371, 1372, 1373, 1375, 1377, 1378, 1494, 
        1495, 1498, 1502, 1526, 1527, 1529, 1530, 1003, 1051, 1068, 1073, 1079, 
        1080, 1083, 1084, 1085, 1086, 1088, 1089, 1093, 1096, 1131, 1168, 1177, 
        1179, 1180, 1190, 1191, 1200, 1237, 1249, 1250, 1507, 1510, 1511, 1512, 
        1549, 1553, 1557, 1558, 1559, 1560, 1561, 1563, 1564, 1576, 1578, 1593, 
        1594, 1598, 1613, 1614, 1615, 1617, 1618, 1620, 1621, 1622, 1624, 1625, 
        1626, 1627, 1629, 1630, 1631, 1632, 1634, 1635, 1786, 1799, 1802, 1804, 
        1806, 1807, 1808, 1809, 1810, 1811, 1813, 1814, 1816, 1818, 1819, 1825, 
        1827, 1830, 1835, 1838, 1849, 1852, 1853, 1854, 1855, 1866, 1873, 1908, 
        1909, 1910, 1911, 1912, 1913, 1915, 1916, 1917, 1918, 1919, 1922, 1924, 
        1925, 1926, 1927, 1928, 1930, 1931, 1932, 1933, 1936, 1939, 1940, 1941, 
        1996, 1997, 1998, 2040, 2043, 2044, 2045, 2046, 2047, 2048, 2096, 2106, 
        2107, 2109, 2110, 2111, 2169, 2170, 2171, 2172
    ]
    
    # Process each drug
    for drug_id in drug_ids:
        try:
            results = process_drug(drug_id, knn)
            if results:
                save_drug_results(drug_id, results)
        except Exception as e:
            print(f"Error processing drug {drug_id}: {str(e)}")
            continue


if __name__ == '__main__':
    main()