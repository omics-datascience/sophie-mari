import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle
from feature_selection import run_ga
from OG_data_preprocessing import create_matrix_and_ic50_for_drug, combined_gdsc, ccle_data


# Train KNN model with selected features using Monte Carlo Cross-Validation
def train_knn_with_mc_cv(X, y, selected_features, n_iter=100, k_neighbors=3):
    """
    Train a KNN model with Monte Carlo cross-validation for multiple iterations.

    Parameters:
        X (pd.DataFrame): Gene expression data matrix (rows = genes, columns = cell lines).
        y (np.array): IC50 values.
        selected_features (list): List of selected feature indices for training.
        n_iter (int): Number of Monte Carlo iterations for cross-validation.
        k_neighbors (int): Number of neighbors to use in KNN (default is 3).

    Returns:
        tuple: (avg_train_preds, avg_test_preds, mse)
            - avg_train_preds (dict): A dictionary containing averaged predicted IC50 values for each sample.
            - avg_test_preds (dict): A dictionary containing averaged predicted IC50 values for each sample in the test set.
            - mse (float): Mean squared error of the predictions on the test set.
    """
    predictions_train = {i: [] for i in range(len(y))}
    predictions_test = {i: [] for i in range(len(y))}
    
    for i in range(n_iter):
        # Split the data into training and testing sets (90% train, 10% test)
        X_train, X_test, y_train, y_test = train_test_split(X.iloc[:, selected_features], y, test_size=0.1, random_state=i)
        
        # Initialize the KNN regressor
        knn = KNeighborsRegressor(n_neighbors=k_neighbors)
        
        # Train the KNN model
        knn.fit(X_train, y_train)
        
        # Predict IC50 for both training and testing sets
        y_train_pred = knn.predict(X_train)
        y_test_pred = knn.predict(X_test)
        
        # Store the predictions
        for idx, pred in zip(range(len(y_train)), y_train_pred):
            predictions_train[idx].append(pred)
        for idx, pred in zip(range(len(y_test)), y_test_pred):
            predictions_test[idx].append(pred)
    
    # Average the predictions over the 100 iterations
    avg_train_preds = {k: np.mean(v) for k, v in predictions_train.items()}
    avg_test_preds = {k: np.mean(v) for k, v in predictions_test.items()}

    # Calculate MSE on the test set (average predictions across iterations)
    mse = mean_squared_error(list(avg_test_preds.values()), y)
    
    return avg_train_preds, avg_test_preds, mse

# Save the trained model
def save_model(model, filename="knn_model.pkl"):
    """
    Save the trained KNN model to a file using pickle.

    Parameters:
        model (KNeighborsRegressor): Trained KNN model.
        filename (str): The name of the file to save the model.
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

# Load the saved model
def load_model(filename="knn_model.pkl"):
    """
    Load a trained KNN model from a file.

    Parameters:
        filename (str): The name of the file to load the model from.

    Returns:
        KNeighborsRegressor: The loaded KNN model.
    """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

# Main function to run feature selection and train the model
def main():
    # Load the data
    X, y =  create_matrix_and_ic50_for_drug(1, combined_gdsc, ccle_data)  ## CHANGE DRUG ID!!! Iterate!

    # Run Genetic Algorithm to perform feature selection
    print("Running Genetic Algorithm for feature selection...")
    best_features, best_mse = run_ga(X, y)  # Make sure this function returns best_features
    print(f"Best selected features: {best_features}")
    print(f"Best MSE: {best_mse}")

    # Train the KNN model with selected features using Monte Carlo Cross-Validation
    print("Training KNN model with selected features...")
    avg_train_preds, avg_test_preds, mse = train_knn_with_mc_cv(X, y, selected_features=best_features)
    print(f"Trained KNN Model - MSE on test set: {mse}")

    # Save the trained model
    model = KNeighborsRegressor(n_neighbors=3)
    model.fit(X.iloc[:, best_features], y)  # Train the final model on selected features
    save_model(model)

if __name__ == "__main__":
    main()
