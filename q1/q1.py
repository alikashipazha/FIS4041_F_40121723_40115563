import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- A) Data Loading and Preprocessing ---

# Load dataset
df = pd.read_csv(r'q1\\loan_train.csv')

# Drop NaNs
df = df.dropna()

# Handling 'Dependents' specifically (usually '3+' causes issues, converting to int)
if 'Dependents' in df.columns:
    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

# Label Encoding for categorical variables
categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Area', 'Status']

le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object' or col in categorical_cols:
        df[col] = le.fit_transform(df[col])

print("Data Preprocessing Completed.")
print(f"Dataset Shape after dropping NaNs: {df.shape}")

# Split Features and Target
X = df.drop(columns=['Status'])
y = df['Status']
feature_names = X.columns.tolist()

# Train/Test Split (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")
print("-" * 30)

# --- Define Objective/Fitness Function ---

def calculate_fitness(mask, alpha=0.9):
    # اگر هیچ ویژگی انتخاب نشده باشد
    if np.sum(mask) == 0:
        # اصلاحیه: باید ۳ مقدار برگرداند تا با خط فراخوانی هماهنگ باشد
        # (Cost=1.0, Accuracy=0.0, Features=0)
        return 1.0, 0.0, 0  
    
    selected_indices = np.where(mask == 1)[0]
    
    # استفاده از نام متغیرهایی که در کدهای قبلی تعریف کردید (X_train, X_test, y_train, ...)
    X_train_sel = X_train.iloc[:, selected_indices]
    X_test_sel = X_test.iloc[:, selected_indices]
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1) #-1 || 1
    clf.fit(X_train_sel, y_train)
    
    y_pred = clf.predict(X_test_sel)
    acc = accuracy_score(y_test, y_pred)
    
    n_features = X.shape[1]
    n_selected = len(selected_indices)
    
    cost = (alpha * (1 - acc)) + ((1 - alpha) * (n_selected / n_features))
    
    return cost, acc, n_selected

import pyswarms as ps

# --- B) Particle Swarm Optimization (PSO) ---

# PSO Parameters
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 50, 'p': 2} # k, p are specific to pyswarms discrete topology
n_particles = 50
iterations = 100
dimensions = X.shape[1] # Number of features

def pso_objective_function(particles, alpha=0.9):
    """
    Pyswarms expects a function that takes (n_particles, dimensions)
    and returns (n_particles, ) cost array.
    """
    n_particles = particles.shape[0]
    costs = []
    
    for i in range(n_particles):
        mask = particles[i]
        cost, _, _ = calculate_fitness(mask, alpha)
        costs.append(cost)
        
    return np.array(costs)

print("Starting PSO...")

# Initialize Binary PSO
optimizer = ps.discrete.binary.BinaryPSO(n_particles=n_particles, dimensions=dimensions, options=options)

# Perform optimization
cost_history_pso, pos_history_pso = optimizer.optimize(pso_objective_function, iters=iterations, verbose=True)

# Extract best solution
best_cost_pso = cost_history_pso
best_pos_pso = pos_history_pso
# Get accuracy and feature count of the best solution
_, acc_pso, num_feat_pso = calculate_fitness(best_pos_pso)
selected_indices_pso = np.where(best_pos_pso == 1)[0]
selected_features_pso = [feature_names[i] for i in selected_indices_pso]

print(f"\nPSO Best Cost: {best_cost_pso:.4f}")
print(f"PSO Final Accuracy: {acc_pso:.4f}")
print(f"PSO Selected Features Count: {num_feat_pso}")
print("-" * 30)

from deap import base, creator, tools, algorithms
import random

# --- C) Genetic Algorithm (GA) ---

# GA Parameters
POPULATION_SIZE = 50
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 50

# 1. Setup DEAP
# We want to minimize the Cost function defined earlier
if hasattr(creator, "FitnessMin"):
    del creator.FitnessMin
if hasattr(creator, "Individual"):
    del creator.Individual

creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # Minimize cost
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator: random binary (0 or 1)
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=dimensions)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluation function
def eval_one_max(individual):
    mask = np.array(individual)
    cost, _, _ = calculate_fitness(mask, alpha=0.9)
    return (cost,)

toolbox.register("evaluate", eval_one_max)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# 2. Run GA
print("Starting GA...")
random.seed(42)
pop = toolbox.population(n=POPULATION_SIZE)
hof = tools.HallOfFame(1) # Store best individual

# Statistics to record
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)

# Run standard GA algo
pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, 
                                   ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=False)

# Extract best solution
best_ind_ga = hof[0]
best_mask_ga = np.array(best_ind_ga)
_, acc_ga, num_feat_ga = calculate_fitness(best_mask_ga)
selected_indices_ga = np.where(best_mask_ga == 1)[0]
selected_features_ga = [feature_names[i] for i in selected_indices_ga]

# Extract history for plotting
min_fitness_values_ga = logbook.select("min")

print(f"\nGA Best Cost: {best_ind_ga.fitness.values[0]:.4f}")
print(f"GA Final Accuracy: {acc_ga:.4f}")
print(f"GA Selected Features Count: {num_feat_ga}")
print("-" * 30)

# --- D & E) Reporting and Analysis ---

print("=== FINAL REPORT ===")
print(f"{'Algorithm':<10} | {'Accuracy':<10} | {'Num Features':<15} | {'Cost':<10}")
print("-" * 55)
print(f"{'PSO':<10} | {acc_pso:<10.4f} | {num_feat_pso:<15} | {best_cost_pso:<10.4f}")
print(f"{'GA':<10} | {acc_ga:<10.4f} | {num_feat_ga:<15} | {best_ind_ga.fitness.values[0]:<10.4f}")

print("\n--- Selected Features by PSO ---")
print(selected_features_pso)

print("\n--- Selected Features by GA ---")
print(selected_features_ga)

# --- F) Visualization and Analysis ---

plt.figure(figsize=(10, 6))

# Plot PSO History
# optimizer.cost_history stores cost per iteration
plt.plot(optimizer.cost_history, label='PSO Cost History', color='blue', linewidth=2)

# Plot GA History
# min_fitness_values_ga stores best cost per generation
plt.plot(min_fitness_values_ga, label='GA Best Cost History', color='red', linewidth=2, linestyle='--')

plt.title('Comparison of PSO and GA Convergence (Feature Selection)')
plt.xlabel('Iterations / Generations')
plt.ylabel('Cost Value (Lower is Better)')
plt.legend()
plt.grid(True)
plt.show()
