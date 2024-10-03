import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    area_columns = [col for col in df.columns if 'area' in col.lower()]
    
    if not area_columns:
        raise ValueError("No area columns found in the CSV file.")
    
    df_clean = df[area_columns].apply(pd.to_numeric, errors='coerce')
    df_clean = df_clean.dropna(axis=1, how='all')
    
    if df_clean.empty:
        raise ValueError("No valid numeric data left after cleaning.")

    imputer = SimpleImputer(strategy='mean')
    df_clean = pd.DataFrame(imputer.fit_transform(df_clean), columns=df_clean.columns)
    
    return df_clean

def fit_slr_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def plot_regression(X, y, model, outliers):
    X_flat = X.flatten()
    y_flat = y.flatten()
    
    plt.figure(facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    plt.scatter(X_flat[~outliers], y_flat[~outliers], color='cyan', label='Data', edgecolor='black')
    
    X_range = np.linspace(X_flat.min(), X_flat.max(), 100).reshape(-1, 1)
    y_pred = model.predict(X_range)
    
    plt.plot(X_range, y_pred, color='red', label='Regression Line')
    
    plt.xlabel('Age', color='black')
    plt.ylabel('Area Measurement', color='black')
    plt.title('Age vs. Area Measurement with Regression Line', color='black')
    plt.tick_params(axis='both', colors='black')
    
    plt.legend()
    plt.show()

def calculate_correlation(X, y):
    correlation_matrix = np.corrcoef(X.flatten(), y.flatten())
    return correlation_matrix[0, 1]

def calculate_residuals(X, y, model):
    predictions = model.predict(X)
    residuals = np.abs(predictions - y)
    return residuals

# Example dictionary with file paths as keys and ages as values
subjects_data = {
    "subject1": {"file_path": "/Users/dhruvsdoc/subject00005_area_mm2.csv", "age": 75},
    "subject2": {"file_path": "/Users/dhruvsdoc/Library/Mobile Documents/com~apple~Numbers/Documents/subject00006_area.csv", "age": 79},
    "subject3": {"file_path": "subject00007_area.csv", "age": 75},
    "subject4": {"file_path": "subject00008_area.csv", "age": 70},
    "subject5": {"file_path": "subject00009_area.csv", "age": 73},
    "subject6": {"file_path": "subject00010_area.csv", "age": 75},
    "subject7": {"file_path": "subject00011_area.csv", "age": 74},
    "subject8": {"file_path": "subject00012_area.csv", "age": 73},
    "subject9": {"file_path": "subject00014_area.csv", "age": 78},
    "subject10": {"file_path": "subject00015_area.csv", "age": 65},
    "subject11": {"file_path": "subject00016_area.csv", "age": 75},
    "subject12": {"file_path": "subject00017_area.csv", "age": 76}
}

X_all = []
y_all = []

for subject, data in subjects_data.items():
    file_path = data['file_path']
    age = data['age']
    
    try:
        df_clean = load_data(file_path)
        print(f"Loaded data for {subject}:")
        print(df_clean.head())  # Print the first few rows for verification
        
        if not df_clean.empty:
            # Use only the first area column for simplicity
            area_measurements = df_clean.iloc[:, 0].values
            
            # Filter out area measurements greater than 700
            area_measurements = area_measurements[area_measurements <= 700]
            
            if len(area_measurements) > 0:
                X_all.extend([age] * len(area_measurements))
                y_all.extend(area_measurements)
            else:
                print(f"Warning: {subject} has no area measurements <= 700.")
        else:
            print(f"Warning: {subject} contains no valid data after cleaning.")
    except ValueError as e:
        print(f"Skipping {subject}: {e}")

# Convert lists to numpy arrays for sklearn
X_all = np.array(X_all).reshape(-1, 1)
y_all = np.array(y_all).astype(float)

# Check if the arrays are empty before fitting the model
if X_all.size == 0 or y_all.size == 0:
    raise ValueError("No valid data found across all subjects. Cannot fit the model.")

# Fit the SLR model with Age as the predictor and Area Measurement as the target
model = fit_slr_model(X_all, y_all)

# Calculate residuals and filter outliers
residuals = calculate_residuals(X_all, y_all, model)
threshold = np.percentile(residuals, 90)  # Adjust the percentile as needed to filter outliers
outliers = residuals > threshold

# Plot the results with the updated function
plot_regression(X_all, y_all, model, outliers)

# Calculate and print the correlation coefficient
correlation = calculate_correlation(X_all, y_all)
print(f"Correlation coefficient: {correlation:.2f}")

# Print model parameters and observations
print(f"Regression Line Parameters:")
print(f"Slope: {model.coef_[0][0]}")
print(f"Intercept: {model.intercept_[0]}")

print(f"Age (x-axis) starts from: {X_all.min()}")
print(f"Age (x-axis) ends at: {X_all.max()}")
print(f"Area Measurement (y-axis) min: {y_all.min()}")
print(f"Area Measurement (y-axis) max: {y_all.max()}")