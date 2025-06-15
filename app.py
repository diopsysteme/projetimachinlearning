from flask import Flask, render_template, jsonify
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
os.makedirs('static/images', exist_ok=True)

# Variable globale pour stocker les résultats
model_results = None
processing_status = "idle"  # idle, processing, completed

def run_classification():
    global processing_status
    processing_status = "processing"
    
    np.random.seed(42)
    n_samples = 10000
    data = {
        'MI_dir_L5_weight': np.random.normal(0.5, 0.2, n_samples),
        'MI_dir_L3_weight': np.random.normal(0.3, 0.15, n_samples),
        'HH_L5_weight': np.random.normal(0.4, 0.18, n_samples),
        'HH_L3_weight': np.random.normal(0.35, 0.12, n_samples),
        'HH_L1_weight': np.random.normal(0.25, 0.1, n_samples),
        'HpHp_L5_weight': np.random.normal(0.45, 0.16, n_samples),
        'HpHp_L3_weight': np.random.normal(0.3, 0.14, n_samples),
        'HpHp_L1_weight': np.random.normal(0.2, 0.08, n_samples),
        'flow_duration': np.random.exponential(2, n_samples),
        'packet_count': np.random.poisson(10, n_samples),
        'byte_count': np.random.exponential(1000, n_samples),
    }

    df = pd.DataFrame(data)
    labels = ['Benign'] * 6000 + ['Mirai'] * 2500 + ['Gafgyt'] * 1500
    np.random.shuffle(labels)
    df['label'] = labels

    for i in range(len(df)):
        if df.iloc[i]['label'] == 'Mirai':
            df.iloc[i, df.columns.get_loc('MI_dir_L5_weight')] += np.random.normal(0.3, 0.1)
            df.iloc[i, df.columns.get_loc('flow_duration')] += np.random.exponential(3)
        elif df.iloc[i]['label'] == 'Gafgyt':
            df.iloc[i, df.columns.get_loc('HH_L5_weight')] += np.random.normal(0.4, 0.1)
            df.iloc[i, df.columns.get_loc('packet_count')] += np.random.poisson(15)

    X = df.drop('label', axis=1)
    y = df['label']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        results[name] = {
            'accuracy': accuracy_score(y_test, preds),
            'f1': f1_score(y_test, preds, average='weighted'),
            'cm': confusion_matrix(y_test, preds),
            'report': classification_report(y_test, preds, output_dict=True)
        }

    best_model_name = 'Random Forest'

    param_grids = {
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5, 10]
        }
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grids[best_model_name],
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)

    optimized_preds = grid_search.predict(X_test_scaled)
    optimized_accuracy = accuracy_score(y_test, optimized_preds)
    optimized_f1 = f1_score(y_test, optimized_preds, average='weighted')
    optimized_cm = confusion_matrix(y_test, optimized_preds)

    results[f'{best_model_name} (Optimisé)'] = {
        'accuracy': optimized_accuracy,
        'f1': optimized_f1,
        'cm': optimized_cm,
        'report': classification_report(y_test, optimized_preds, output_dict=True)
    }

    # Comparaison
    comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [v['accuracy'] for v in results.values()],
        'F1-Score': [v['f1'] for v in results.values()]
    })

    # Graphiques
    plt.figure(figsize=(10, 5))
    x_pos = np.arange(len(comparison))
    plt.bar(x_pos, comparison['Accuracy'], alpha=0.8, color=['#3498db', '#e74c3c', '#f39c12', '#27ae60'])
    plt.xlabel('Modèles')
    plt.ylabel('Précision')
    plt.title('Comparaison de la Précision des Modèles')
    plt.xticks(x_pos, comparison['Model'], rotation=45)
    plt.ylim(0, 1)
    for i, v in enumerate(comparison['Accuracy']):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('static/images/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x_pos, comparison['F1-Score'], alpha=0.8, color=['#9b59b6', '#34495e', '#16a085', '#e67e22'])
    plt.xlabel('Modèles')
    plt.ylabel('Score F1')
    plt.title('Comparaison du Score F1 des Modèles')
    plt.xticks(x_pos, comparison['Model'], rotation=45)
    plt.ylim(0, 1)
    for i, v in enumerate(comparison['F1-Score']):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('static/images/f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Génération des matrices de confusion individuelles
    for model_name, model_data in results.items():
        plt.figure(figsize=(6, 5))
        sns.heatmap(model_data['cm'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'Matrice de Confusion - {model_name}')
        plt.xlabel('Prédictions')
        plt.ylabel('Réalité')
        plt.tight_layout()
        clean_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('é', 'e')
        plt.savefig(f'static/images/cm_{clean_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Matrice globale
    plt.figure(figsize=(8, 6))
    sns.heatmap(results['Random Forest (Optimisé)']['cm'], annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Matrice de Confusion - Meilleur Modèle')
    plt.xlabel('Prédictions')
    plt.ylabel('Réalité')
    plt.tight_layout()
    plt.savefig('static/images/all_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Importance des caractéristiques (Random Forest)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Importance')
    plt.title('Importance des Caractéristiques (Random Forest)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('static/images/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    processing_status = "completed"
    return comparison.round(3).to_dict(orient='records')

@app.route("/")
def index():
    global model_results, processing_status
    return render_template("index.html", results=model_results, status=processing_status)

@app.route("/start_processing")
def start_processing():
    global model_results
    model_results = run_classification()
    return jsonify({"status": "completed", "results": model_results})

@app.route("/status")
def get_status():
    return jsonify({"status": processing_status})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
