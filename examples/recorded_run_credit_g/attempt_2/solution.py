import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegressionCV

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

X_train_trans = preprocessor.fit_transform(X_train)
X_val_trans = preprocessor.transform(X_val)
X_test_trans = preprocessor.transform(X_test)

# Hyperparameter tuning for C using LogisticRegressionCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
Cs = np.logspace(-3, 2, 40)

lr_cv = LogisticRegressionCV(
    Cs=Cs,
    cv=cv,
    scoring='neg_log_loss',
    penalty='l2',
    max_iter=1000,
    random_state=0
)

lr_cv.fit(X_train_trans, y_train)

# Extract best CV score
pos_class = lr_cv.classes_[1]
mean_scores_per_c = lr_cv.scores_[pos_class].mean(axis=0)
best_cv_score = np.max(mean_scores_per_c)
print(f"CV negative log-loss: {best_cv_score:.4f}")

# Predict on validation and test sets
val_probs = lr_cv.predict_proba(X_val_trans)
test_probs = lr_cv.predict_proba(X_test_trans)

val_probs = np.clip(val_probs, 1e-6, 1 - 1e-6)
test_probs = np.clip(test_probs, 1e-6, 1 - 1e-6)

classes = lr_cv.classes_
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