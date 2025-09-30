# Problem 1

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

def train_basic_decision_tree(df, le_drug):
    X = df[['Age', 'Sex', 'BP', 'Cholesterol']]
    y = df['Drug']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)

    print("=== Model A: Basic Decision Tree ===")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le_drug.classes_))

    plt.figure(figsize=(20, 10))  
    plot_tree(
        clf,
        feature_names=['Age', 'Sex', 'BP', 'Cholesterol'],
        class_names=le_drug.classes_,
        filled=True,
        rounded=True,
        max_depth=3  
    )
    plt.title("Model A: Basic Decision Tree (Top 3 Levels)")
    plt.tight_layout()
    plt.savefig("output/decision_tree_basic.png", dpi=300)
    plt.close()

def train_smote_decision_tree(df, le_drug):
    X = df.drop(columns=['Drug'])
    y = df['Drug']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    clf = DecisionTreeClassifier(random_state=42, max_depth=5)
    clf.fit(X_train_smote, y_train_smote)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n=== Model B: Decision Tree + SMOTE ===")
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le_drug.classes_))
    plt.figure(figsize=(14, 8))
    plot_tree(clf, feature_names=X.columns,
              class_names=le_drug.classes_, filled=True)
    plt.title("Model B: Decision Tree with SMOTE")
    plt.tight_layout()
    plt.savefig("output/decision_tree_smote.png")
    plt.close()
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_drug.classes_)
    plt.figure(figsize=(6, 6))
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Model B: Confusion Matrix")
    plt.savefig("output/confusion_matrix_smote.png")
    plt.close()
    importances = clf.feature_importances_
    plt.figure(figsize=(8, 5))
    plt.barh(X.columns, importances)
    plt.xlabel("Feature Importance")
    plt.title("Model B: Feature Importance (with SMOTE)")
    plt.tight_layout()
    plt.savefig("output/feature_importance_smote.png")
    plt.close()

def run_problem1():
    df = pd.read_csv("drug200.csv")
    os.makedirs("output", exist_ok=True)
    label_encoders = {}
    for col in ['Sex', 'BP', 'Cholesterol', 'Drug']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    train_basic_decision_tree(df.copy(), label_encoders['Drug'])
    train_smote_decision_tree(df.copy(), label_encoders['Drug'])

run_problem1()

# Problem 2-LR

import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("output2", exist_ok=True)
df = pd.read_csv("default of credit card clients.csv")
df = df.drop(columns=['ID'])
X = df.drop(columns=['dpnm'])
y = df['dpnm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train_scaled, y_train)
y_pred_base = baseline_model.predict(X_test_scaled)
y_proba_base = baseline_model.predict_proba(X_test_scaled)
print("=== Baseline Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_base))
print("Log Loss:", log_loss(y_test, y_proba_base))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_base))
print("Classification Report:\n", classification_report(y_test, y_pred_base))

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000))
])
param_grid = {
    'lr__C': [0.01, 0.1, 1, 10],
    'lr__penalty': ['l2'],
    'lr__solver': ['lbfgs', 'liblinear']
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='recall', n_jobs=-1)
grid.fit(X_train, y_train)
print("=== GridSearchCV Logistic Regression ===")
print("Best Parameters:", grid.best_params_)
best_model = grid.best_estimator_
y_pred_grid = best_model.predict(X_test)
y_proba_grid = best_model.predict_proba(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_grid))
print("Log Loss:", log_loss(y_test, y_proba_grid))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_grid))
print("Classification Report:\n", classification_report(y_test, y_pred_grid))
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_grid), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred 0', 'Pred 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (GridSearchCV)')
plt.tight_layout()
plt.savefig("output2/lr_grid_confusion_matrix.png", dpi=300)
plt.show()

# Problem 2-DT

from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv("default of credit card clients.csv")
df = df.drop(columns=['ID'])
X = df.drop(columns=['dpnm'])
y = df['dpnm']
feature_names = X.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred_a = model.predict(X_test)
y_proba_a = model.predict_proba(X_test)
acc_a = accuracy_score(y_test, y_pred_a)
loss_a = log_loss(y_test, y_proba_a)
cm_a = confusion_matrix(y_test, y_pred_a)
report_a = classification_report(y_test, y_pred_a)
print("=== Basic Decision Tree ===")
print("Accuracy:", acc_a)
print("Log Loss:", loss_a)
print("Confusion Matrix:\n", cm_a)
print("Classification Report:\n", report_a)
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=feature_names, class_names=['No Default', 'Default'], filled=True, rounded=True, max_depth=5)
plt.title("Basic Decision Tree (Top 3 Levels)")
plt.savefig("output2/basic_decision_tree.png", dpi=300, bbox_inches='tight')
plt.show()

param_grid = {
    'max_depth': [3, 5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print("\n=== Tuned Decision Tree ===")
print("Best Parameters:", grid.best_params_)
best_model = grid.best_estimator_
y_pred_b = best_model.predict(X_test)
y_proba_b = best_model.predict_proba(X_test)
acc_b = accuracy_score(y_test, y_pred_b)
loss_b = log_loss(y_test, y_proba_b)
cm_b = confusion_matrix(y_test, y_pred_b)
report_b = classification_report(y_test, y_pred_b)
print("Accuracy:", acc_b)
print("Log Loss:", loss_b)
print("Confusion Matrix:\n", cm_b)
print("Classification Report:\n", report_b)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_b, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred 0', 'Pred 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Tuned Decision Tree')
plt.tight_layout()
plt.savefig("output2/tuned_dt_confusion_matrix.png", dpi=300)
plt.show()
plt.figure(figsize=(20, 10))
plot_tree(best_model, feature_names=feature_names, class_names=['No Default', 'Default'], filled=True, rounded=True, max_depth=3)
plt.title("Tuned Decision Tree (Top 3 Levels)")
plt.savefig("output2/tuned_decision_tree.png", dpi=300, bbox_inches='tight')
plt.show()

# Problem 2-NN

from sklearn.neural_network import MLPClassifier

df = pd.read_csv("default of credit card clients.csv")
df = df.drop(columns=['ID'])
X = df.drop(columns=['dpnm'])
y = df['dpnm']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

baseline_mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42, verbose=False, early_stopping=False)
baseline_mlp.fit(X_train, y_train)
y_pred_base = baseline_mlp.predict(X_test)
y_proba_base = baseline_mlp.predict_proba(X_test)
print("=== Baseline Neural Network ===")
print("Accuracy:", accuracy_score(y_test, y_pred_base))
print("Log Loss:", log_loss(y_test, y_proba_base))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_base))
print("Classification Report:\n", classification_report(y_test, y_pred_base))

param_grid = {
    'hidden_layer_sizes': [(100,), (100, 50)],
    'alpha': [0.0001, 0.001],
    'learning_rate_init': [0.001, 0.01],
    'solver': ['adam']
}
grid = GridSearchCV(
    MLPClassifier(max_iter=300, early_stopping=True, random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)
grid.fit(X_train, y_train)
print("\n=== Tuned Neural Network ===")
print("Best Parameters:", grid.best_params_)
best_model = grid.best_estimator_
y_pred_grid = best_model.predict(X_test)
y_proba_grid = best_model.predict_proba(X_test)
accuracy = accuracy_score(y_test, y_pred_grid)
loss = log_loss(y_test, y_proba_grid)
conf_matrix = confusion_matrix(y_test, y_pred_grid)
report = classification_report(y_test, y_pred_grid)
print("Accuracy:", accuracy)
print("Log Loss:", loss)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred 0', 'Pred 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Tuned Neural Network)')
plt.tight_layout()
plt.savefig("output2/nn_grid_confusion_matrix.png", dpi=300)
plt.show()

# Problem 3
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("train.csv")
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features].copy()
y = df['Survived']

le_sex = LabelEncoder()
le_embarked = LabelEncoder()
X['Sex'] = le_sex.fit_transform(X['Sex'].astype(str)).astype('int')
X['Embarked'] = le_embarked.fit_transform(X['Embarked'].astype(str)).astype('int')

imputer = SimpleImputer(strategy='mean')
X[['Age', 'Fare']] = imputer.fit_transform(X[['Age', 'Fare']])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    model = XGBClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy").mean()
    return score

print("Starting Optuna hyperparameter tuning...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best parameters:", study.best_params)

best_model = XGBClassifier(**study.best_params)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_val)
print("\nAccuracy:", accuracy_score(y_val, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))

# Save confusion matrix plot
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Validation Set')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

df_test = pd.read_csv("test.csv")
X_test_real = df_test[features].copy()
X_test_real['Sex'] = le_sex.transform(X_test_real['Sex'].astype(str)).astype('int')
X_test_real['Embarked'] = le_embarked.transform(X_test_real['Embarked'].astype(str)).astype('int')
X_test_real[['Age', 'Fare']] = imputer.transform(X_test_real[['Age', 'Fare']])

y_pred_real = best_model.predict(X_test_real)
output = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_pred_real})
output.to_csv("titanic_prediction_optuna_xgb.csv", index=False)
print("Prediction result saved as titanic_prediction_optuna_xgb.csv")
print("Confusion matrix image saved as confusion_matrix.png")
