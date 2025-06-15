# 🔐 Système de Détection d'Intrusions IoT avec Intelligence Artificielle

## 📋 Description

Ce projet implémente un **système intelligent de cybersécurité** utilisant des algorithmes de Machine Learning pour détecter automatiquement les intrusions dans les réseaux d'objets connectés (IoT). Le système est capable d'identifier les attaques **Mirai** et **Gafgyt**, deux des principales menaces ciblant les appareils IoT.
  ### LIEN EN LIGNE  https://projetia-lcfq.onrender.com/ & SUR DOCKERHUB https://hub.docker.com/repository/docker/diopsysteme/projetia/

### 🎯 Objectifs
- Détection automatique des cyber-attaques IoT en temps réel
- Classification multi-classes : Trafic Normal, Attaque Mirai, Attaque Gafgyt
- Interface web interactive pour visualiser les résultats
- Comparaison de performances entre différents algorithmes IA

## 🚀 Fonctionnalités

### 🤖 Modèles IA Implémentés
- **Random Forest** : Forêt d'arbres de décision pour une classification robuste
- **SVM (Support Vector Machine)** : Classification par séparation optimale des classes
- **Régression Logistique** : Approche probabiliste rapide et interprétable
- **Optimisation automatique** : GridSearchCV pour l'ajustement des hyperparamètres

### 📊 Visualisations
- Comparaison des performances (Précision, F1-Score)
- Matrices de confusion détaillées
- Importance des caractéristiques
- Graphiques interactifs générés automatiquement

### 🌐 Interface Web
- Interface utilisateur intuitive avec Flask
- Traitement asynchrone des données
- Visualisation en temps réel des résultats
- Design responsive et moderne

## 🛠️ Technologies Utilisées

### Backend
- **Flask** : Framework web Python
- **scikit-learn** : Algorithmes de Machine Learning
- **pandas** : Manipulation des données
- **numpy** : Calculs numériques
- **matplotlib/seaborn** : Génération de graphiques

### Frontend
- **HTML5/CSS3** : Interface utilisateur moderne
- **JavaScript** : Interactions dynamiques
- **Design responsive** : Compatible mobile/desktop

## 📦 Installation

### Prérequis
- Python 3.7+
- pip (gestionnaire de paquets Python)

### 1. Cloner le repository
```bash
git clone https://github.com/votre-username/iot-intrusion-detection.git
cd iot-intrusion-detection
```

### 2. Créer un environnement virtuel (recommandé)
```bash
python -m venv venv

perso je suis sur LINUX

### 3. Installer les dépendances
```bash
pip install requirements.txt
```

### 4. Structure des fichiers
```
projetia/
│
├── app.py                 # Application Flask principale
├── templates/
│   └── index.html        # Interface web
├── static/
│   └── images/           # Graphiques générés (créé automatiquement)
└── README.md             # Ce fichier
└── requirements.txt  
```

## 🚀 Utilisation
### 0. LIEN EN LIGNE  https://projetia-lcfq.onrender.com/ & SUR DOCKERHUB https://hub.docker.com/repository/docker/diopsysteme/projetia/
### 0.  POUR LE RUN PUL ET RUN 
```bash
docker pull diopsysteme/projetia 
docker run -p 5000:5000  diopsysteme/projetia 
``` 
L'appli est deployée sur RENDER

### 1. Démarrer l'application
```bash
python app.py
```
et l'appli demarrera sur le port 5000
    

### 2. Accéder à l'interface web
Ouvrez votre navigateur et allez à : `http://localhost:5000`

### 3. Lancer l'analyse
1. Cliquez sur **"Démarrer l'Analyse"**
2. Attendez que l'entraînement des modèles se termine
3. Consultez les résultats et visualisations générés

## 📈 Données et Caractéristiques

### 🔍 Caractéristiques Analysées
- **MI_dir_L5_weight** : Poids directionnel Mutual Information niveau 5
- **MI_dir_L3_weight** : Poids directionnel Mutual Information niveau 3
- **HH_L5_weight** : Poids Header-Header niveau 5
- **HH_L3_weight** : Poids Header-Header niveau 3
- **HH_L1_weight** : Poids Header-Header niveau 1
- **HpHp_L5_weight** : Poids Header-Payload niveau 5
- **HpHp_L3_weight** : Poids Header-Payload niveau 3
- **HpHp_L1_weight** : Poids Header-Payload niveau 1
- **flow_duration** : Durée des flux réseau
- **packet_count** : Nombre de paquets
- **byte_count** : Nombre d'octets

### 📊 Dataset
- **10,000 échantillons** générés de manière réaliste
- **Distribution** : 60% Benign, 25% Mirai, 15% Gafgyt
- **Caractéristiques d'attaques** injectées automatiquement

## ⚙️ Configuration

### Paramètres d'optimisation (GridSearchCV)
```python
param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5, 10]
    }
}
```

### Métriques d'évaluation
- **Accuracy** : Précision globale
- **F1-Score** : Moyenne harmonique précision/rappel
- **Confusion Matrix** : Analyse détaillée des classifications
- **Classification Report** : Rapport complet par classe

## 📊 Résultats Attendus

Les modèles atteignent généralement :
- **Random Forest** : ~85-95% de précision
- **SVM** : ~80-90% de précision
- **Régression Logistique** : ~75-85% de précision

L'optimisation par GridSearch améliore typiquement les performances de 2-5%.

## 🔧 Endpoints API

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Page principale |
| `/start_processing` | GET | Lance l'analyse ML |
| `/status` | GET | Statut du traitement |

## 🎓 Contexte Éducatif

Ce projet est conçu comme un **TP (Travail Pratique)** pour démontrer :
- L'application de l'IA en cybersécurité
- La détection d'intrusions dans l'IoT
- La comparaison d'algorithmes ML
- Le développement d'interfaces web interactives

### 📚 Concepts Couverts
- **Machine Learning supervisé**
- **Classification multi-classes**
- **Optimisation d'hyperparamètres**
- **Validation croisée**
- **Métriques d'évaluation**
- **Visualisation de données**

## 🔒 Sécurité et Limitations

### ⚠️ Limitations
- Données simulées (non issues de vrais réseaux IoT)
- Modèles entraînés sur un dataset spécifique
- Performance peut varier sur données réelles

### 🛡️ Améliorations Possibles
- Intégration de données réseau réelles
- Ajout d'autres types d'attaques
- Implémentation de la détection en temps réel
- Déploiement sur infrastructure cloud



Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.


