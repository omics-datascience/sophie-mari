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

    # Debug check: Make sure all features exist in X
    for f in feature_names:
        if f not in X.columns:
            print(f"❌ Feature not in X.columns: {f}")
            print("Available columns sample:", X.columns[:10].tolist())
            raise ValueError(f"Feature {f} not found in X.columns")
    
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


# Automatically set project root, so that we can both run the code and save results in sophie-mari
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GDSC_CSV_PATH = PROJECT_ROOT / "datasets" / "combined_gdsc.csv"

try:
    gdsc_df = pd.read_csv(GDSC_CSV_PATH)
    drug_id_to_name = (
        pd.Series(gdsc_df["DRUG_NAME"].values, index=gdsc_df["DRUG_ID"])
        .dropna()
        .to_dict()
    )
    print(f"✅ Loaded drug name dictionary with {len(drug_id_to_name)} entries")
except Exception as e:
    print(f"⚠️ Could not load drug ID to name mapping: {e}")
    drug_id_to_name = {}


GENE_EXPR_DIR = PROJECT_ROOT / "GENE_EXPR_DIR"
IC50_DIR = PROJECT_ROOT / "IC50_DIR"
RESULTS_DIR = PROJECT_ROOT / "results"

# Old code for defining the exact paths for saving
'''GENE_EXPR_DIR = "/Users/sophiaboler/Desktop/sophie-mari/GENE_EXPR_DIR"
IC50_DIR = "/Users/sophiaboler/Desktop/sophie-mari/IC50_DIR"
RESULTS_DIR = "/Users/sophiaboler/Desktop/sophie-mari/results"'''

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
        y = pd.read_csv(ic50_path, header=None, skiprows=1).squeeze()

        # Align the indices
        if len(y) != X.shape[0]:
            print("❌ Mismatch between IC50 values and gene expression samples!")
            return None, None

        y.index = X.index  # ✅ Make sure they match

        # Z-score normalization (per sample)
        X = X.sub(X.mean(axis=1), axis=0).div(X.std(axis=1), axis=0)

        # Sanity check
        print("🔍 Mean across genes per sample (should be ~0):")
        print(X.mean(axis=1).round(4).head())
        print("🔍 Std across genes per sample (should be ~1):")
        print(X.std(axis=1).round(4).head())

        return X, y
    
    except Exception as e:
        print(f"❌ Error loading drug {drug_id}: {str(e)}")
        return None, None

def process_drug(drug_id: int, model: KNeighborsRegressor) -> Dict:
    """Process a single drug through the full GA/KNN pipeline."""
    X, y = load_drug_data(drug_id)
    if X is None or y is None:
        return None
    
    drug_name = drug_id_to_name.get(drug_id, "Unknown Drug")
    print(f"\nProcessing drug {drug_id} ({drug_name}) with {X.shape[1]} genes and {len(y)} samples")

    # Initialize storage for results
    selected_features_all = []
    test_preds_by_sample = defaultdict(list)
    test_actuals_by_sample = defaultdict(list)

    for split in tqdm(range(NUM_SPLITS), desc=f"Drug {drug_id}"):
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=split
        )

        # Feature selection via GA on training set
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
        selected_features_all.append(features)

        # Predict IC50 for test set using selected features
        X_test_selected = X_test[features]
        y_pred = knn_model.predict(X_test_selected)

        # Store predictions by sample name (index in X)
        for i, sample_name in enumerate(X_test.index):
            test_preds_by_sample[sample_name].append(y_pred[i])
            test_actuals_by_sample[sample_name].append(y_test.loc[sample_name])

    return {
        "drug_id": drug_id,
        "selected_features": selected_features_all,
        "predictions_by_sample": test_preds_by_sample,
        "actuals_by_sample": test_actuals_by_sample
    }

def save_drug_results(drug_id: int, results: Dict):
    drug_dir = Path(RESULTS_DIR) / str(drug_id)
    drug_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results for drug {drug_id} to {drug_dir}")

    try:
        # Save selected features (100 lists of 30)
        features_df = pd.DataFrame(results['selected_features']).T
        features_df.to_csv(drug_dir / "selected_features.csv", index=False)
        print("✅ Saved selected_features.csv")
    except Exception as e:
        print(f"❌ Error saving selected_features.csv: {e}")

    try:
        # Convert defaultdicts to DataFrames
        preds_df = pd.DataFrame.from_dict(results["predictions_by_sample"], orient="index").T
        actuals_df = pd.DataFrame.from_dict(results["actuals_by_sample"], orient="index").T

        preds_df.to_csv(drug_dir / "predictions.csv", index=False)
        actuals_df.to_csv(drug_dir / "actuals.csv", index=False)
        print("✅ Saved predictions.csv and actuals.csv")

        # Final averaged predictions
        final_preds = preds_df.mean(axis=0)
        final_actuals = actuals_df.mean(axis=0)

        final_df = pd.DataFrame({
            "actual": final_actuals,
            "predicted": final_preds
        })
        final_df.to_csv(drug_dir / "final_predictions.csv")
        print("✅ Saved final_predictions.csv")

    except Exception as e:
        print(f"❌ Error saving prediction data: {e}")

    try:
        feature_counts = Counter(f for sublist in results['selected_features'] for f in sublist)
        freq_df = pd.DataFrame.from_dict(feature_counts, orient='index', columns=['count']).sort_values('count', ascending=False)
        freq_df.to_csv(drug_dir / "feature_frequencies.csv")
        print("✅ Saved feature_frequencies.csv")
    except Exception as e:
        print(f"❌ Error saving feature frequencies: {e}")

'''def process_drug(drug_id: int, model: KNeighborsRegressor) -> Dict:
    """Process a single drug through the full pipeline."""
    X, y = load_drug_data(drug_id)
    if X is None or y is None:
        return None
    drug_name = drug_id_to_name.get(drug_id, "Unknown Drug")
    print(f"\nProcessing drug {drug_id} ({drug_name}) with {X.shape[1]} genes and {len(y)} samples")
    
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

        # Store predictions + actuals together
        drug_results['selected_features'].append(features)
        drug_results['predictions'].append((y_pred, y_test.values))  # <-- tuple
        drug_results['models'].append(knn_model)

    
    return drug_results


def save_drug_results(drug_id: int, results: Dict):
    """Save all results for a drug in an organized structure, with debug checks."""
    drug_dir = Path(RESULTS_DIR) / str(drug_id)
    drug_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving results for drug {drug_id} to {drug_dir}")

    try:
        features_df = pd.DataFrame(results['selected_features']).T
        print("✅ features_df shape:", features_df.shape)
        features_df.to_csv(drug_dir / "selected_features.csv")
    except Exception as e:
        print(f"❌ Error saving selected_features.csv: {e}")

    try:
        # Unpack the (y_pred, y_test) tuples
        pred_arrays = []
        actual_arrays = []

        for y_pred, y_test in results['predictions']:
            pred_arrays.append(y_pred)
            actual_arrays.append(y_test)

        preds_df = pd.DataFrame(pred_arrays).T
        actuals_df = pd.DataFrame(actual_arrays).T

        print("✅ preds_df shape:", preds_df.shape)
        print("✅ actuals_df shape:", actuals_df.shape)

        preds_df.to_csv(drug_dir / "predictions.csv", index=False)
        actuals_df.to_csv(drug_dir / "actuals.csv", index=False)
    except Exception as e:
        print(f"❌ Error saving predictions or actuals: {e}")

    try:
        mean_preds = preds_df.mean(axis=1)
        mean_actuals = actuals_df.mean(axis=1)  # Optional averaging for final view

        final_df = pd.DataFrame({
            'actual': mean_actuals,
            'predicted': mean_preds
        })
        print("✅ final_df shape:", final_df.shape)
        final_df.to_csv(drug_dir / "final_predictions.csv", index=False)
    except Exception as e:
        print(f"❌ Error saving final_predictions.csv: {e}")

    try:
        feature_counts = Counter([
            f for sublist in results['selected_features'] for f in sublist
        ])
        freq_df = pd.DataFrame.from_dict(
            feature_counts, orient='index', columns=['count']
        ).sort_values('count', ascending=False)
        print("✅ freq_df shape:", freq_df.shape)
        freq_df.to_csv(drug_dir / "feature_frequencies.csv")
    except Exception as e:
        print(f"❌ Error saving feature_frequencies.csv: {e}")'''

def is_drug_already_processed(drug_id: int) -> bool:
    """Check if results for a drug already exist."""
    expected_file = RESULTS_DIR / str(drug_id) / "final_predictions.csv"
    return expected_file.exists()

def main():
    # Initialize model
    knn = KNeighborsRegressor(n_neighbors=3)
    
    # Load drug IDs
    drug_ids = [1190, 1003, 1088, 1814, 1563, 252, 1931, 1594, 1373, 1079, 1051]
    
    # Process each drug
    for drug_id in drug_ids:
        if is_drug_already_processed(drug_id):
            print(f"⏩ Skipping drug {drug_id} (results already exist)")
            continue

        try:
            results = process_drug(drug_id, knn)
            if results:
                print(f"Saving results for drug {drug_id}")
                save_drug_results(drug_id, results)
        except Exception as e:
            print(f"❌ Error processing drug {drug_id}: {e}")
            continue

    '''# Runs model on specific drug (change it to the ID you want to test)
    drug_id = 1003 

    try:
        results = process_drug(drug_id, knn)
        if results:
            print(f"Saving results for drug {drug_id}")
            save_drug_results(drug_id, results)
    except Exception as e:
        print(f"Error processing drug {drug_id}: {str(e)}")'''
    


if __name__ == '__main__':
    main()