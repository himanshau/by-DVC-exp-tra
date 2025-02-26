import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

# Generate features with standard criteria
num_samples = 1000
IQ = np.random.normal(loc=110, scale=15, size=num_samples).clip(70, 160)
CGPA = np.random.normal(loc=7, scale=1, size=num_samples).clip(0, 10)
marks_10th = np.random.normal(loc=75, scale=10, size=num_samples).clip(50, 100)
marks_12th = np.random.normal(loc=72, scale=10, size=num_samples).clip(50, 100)
communication_skills = np.random.normal(loc=6, scale=2, size=num_samples).clip(1, 10)

# Generate 'Placed' column based on a probability formula
probability = (0.4 * CGPA + 0.3 * (IQ / 160) + 0.3 * (communication_skills / 10))
placed = np.random.binomial(1, probability.clip(0, 1))

# Create a DataFrame
df = pd.DataFrame({
    'IQ': IQ,
    'CGPA': CGPA,
    '10th_marks': marks_10th,
    '12th_marks': marks_12th,
    'communication_skills': communication_skills,
    'placed': placed
})

# Create directory and save dataset
os.makedirs("data", exist_ok=True)
df.to_csv("data/student_performance.csv", index=False)

print("Dataset generated and saved as 'data/student_performance.csv'")
