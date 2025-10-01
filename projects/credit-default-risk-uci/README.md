# Credit Card Default Prediction (UCI Dataset)

This project predicts credit card default using the UCI "Default of Credit Card Clients" dataset (30,000 clients, 24 variables).  
The analysis was conducted as part of coursework at UW–Madison (AAE 718).

## Project Overview
- **Goal:** Predict whether a client will default on their credit card payment.  
- **Data:** UCI "Default of Credit Card Clients" dataset (30,000 clients, 24 features).  
- **Methods:** Logistic Regression, Decision Tree, Neural Network.  
- **Techniques:** GridSearchCV hyperparameter tuning, SMOTE to address class imbalance.  

## Key Results
- Accuracy: **~82%**  
- Default-class precision: **improved from 0.37 → 0.65**  
- Default-class recall: **up to ~40%**  
- Log loss: **0.4446–0.4675**  
- Demonstrated application of **credit risk modeling techniques** beyond accuracy, using precision/recall, log loss, and confusion matrices.  

## Files
- [`credit_default_project.pdf`](./credit_default_project.pdf) – Full report with methods, figures, and analysis discussion.  
- [`11all.py`](./11all.py) – Python code implementing models and evaluations.  

## Requirements
- Python 3.10+  
- Packages: `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `matplotlib`, `seaborn`  

Install with:  
```bash
pip install -r requirements.txt
```

## How to Run
```bash
python 11all.py
```
## Author
Tingyun (Caroline) Kuo
