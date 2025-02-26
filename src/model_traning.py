import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dvclive import Live
# Load dataset
df = pd.read_csv("data/student_performance.csv")

# Define features and target
X = df.drop(columns=['placed'])
y = df['placed']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_estimators = 10
max_depth = 5

# Initialize and train the model
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)



# Print results

with Live(save_dvc_exp=True) as live:
    live.log_metric('Accuracy:',accuracy_score(y_test, y_pred))
    live.log_metric('Precision:',precision_score(y_test, y_pred))
    live.log_metric('F1 Score:',f1_score(y_test, y_pred))
    live.log_metric('Recall: ',recall_score(y_test, y_pred))

    live.log_param('n_estimators:',n_estimators)
    live.log_param('max_depth:',max_depth)