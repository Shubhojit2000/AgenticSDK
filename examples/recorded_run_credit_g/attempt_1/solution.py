import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline

# Load data
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val_features.csv')
test_df = pd.read_csv('test_features.csv')

# Separate features and target
target_col = 'target'
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col].astype(str)

X_val = val_df.copy()
X_test = test_df.copy()

# Identify categorical and numeric columns
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

# Model
# Use L2 regularization with tuned C for well-calibrated probabilities
model = LogisticRegression(C=0.5, penalty='l2', max_iter=1000, random_state=0)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Cross-validation for evaluation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
log_losses = []

for train_idx, val_idx in cv.split(X_train, y_train):
    X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    pipeline.fit(X_tr, y_tr)
    preds = pipeline.predict_proba(X_va)
    # Clip probabilities to match evaluation metric bounds
    preds = np.clip(preds, 1e-6, 1 - 1e-6)
    loss = log_loss(y_va, preds, labels=['bad', 'good'])
    log_losses.append(loss)

mean_neg_log_loss = -np.mean(log_losses)
print(f"CV negative log-loss: {mean_neg_log_loss:.4f}")

# Train on full training set
pipeline.fit(X_train, y_train)

# Predict on validation and test sets
val_probs = pipeline.predict_proba(X_val)
test_probs = pipeline.predict_proba(X_test)

val_probs = np.clip(val_probs, 1e-6, 1 - 1e-6)
test_probs = np.clip(test_probs, 1e-6, 1 - 1e-6)

# Ensure class order is exactly ['bad', 'good']
classes = pipeline.classes_
bad_idx = list(classes).index('bad')
good_idx = list(classes).index('good')

val_preds_df = pd.DataFrame({
    'id': range(len(val_df)),
    'bad': val_probs[:, bad_idx],
    'good': val_probs[:, good_idx]
})

test_preds_df = pd.DataFrame({
    'id': range(len(test_df)),
    'bad': test_probs[:, bad_idx],
    'good': test_probs[:, good_idx]
})

val_preds_df.to_csv('predictions_val.csv', index=False)
test_preds_df.to_csv('predictions_test.csv', index=False)
print("Predictions successfully written.")