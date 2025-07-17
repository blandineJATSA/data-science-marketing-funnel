"""
Module d'analyses avancées pour le marketing
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class MarketingAnalytics:
    """Classe pour les analyses marketing avancées"""
    
    def __init__(self):
        self.models = {}
        self.encoders = {}
    
    def analyze_customer_segments(self, data: pd.DataFrame) -> Dict:
        """
        Analyse complète des segments clients
        
        Args:
            data: DataFrame avec les données clients
            
        Returns:
            Dictionnaire avec les analyses par segment
        """
        segment_analysis = data.groupby('RFM_Segment').agg({
            'user_id': 'count',
            'reached_confirmation': 'sum',
            'monetary_value': ['mean', 'std', 'sum'],
            'clv': ['mean', 'std', 'sum'],
            'churn_probability': ['mean', 'std'],
            'recency': 'mean',
            'frequency': 'mean',
            'engagement_score': 'mean'
        }).round(2)
        
        # Aplatir les colonnes multi-niveaux
        segment_analysis.columns = ['_'.join(col).strip() for col in segment_analysis.columns.values]
        
        # Calculer les métriques additionnelles
        segment_analysis['conversion_rate'] = (
            segment_analysis['reached_confirmation_sum'] / 
            segment_analysis['user_id_count'] * 100
        ).round(2)
        
        segment_analysis['revenue_per_user'] = (
            segment_analysis['monetary_value_sum'] / 
            segment_analysis['user_id_count']
        ).round(2)
        
        # Identifier les segments prioritaires
        segment_analysis['priority_score'] = (
            segment_analysis['conversion_rate'] * 0.3 +
            segment_analysis['clv_mean'] * 0.4 +
            (100 - segment_analysis['churn_probability_mean'] * 100) * 0.3
        ).round(2)
        
        return segment_analysis.to_dict()
    
    def predict_churn(self, data: pd.DataFrame, test_size: float = 0.3) -> Dict:
        """
        Modèle de prédiction du churn
        
        Args:
            data: DataFrame avec les données clients
            test_size: Taille de l'échantillon de test
            
        Returns:
            Dictionnaire avec les résultats du modèle
        """
        # Préparation des features
        feature_columns = [
            'recency', 'frequency', 'monetary_value', 
            'R_Score', 'F_Score', 'M_Score',
            'clv', 'engagement_score', 'page_views', 'session_duration'
        ]
        
        # Encoder les variables catégorielles
        categorical_features = ['device', 'traffic_source', 'country']
        
        X = data[feature_columns].copy()
        
        for cat_feature in categorical_features:
            if cat_feature in data.columns:
                le = LabelEncoder()
                X[f'{cat_feature}_encoded'] = le.fit_transform(data[cat_feature])
                self.encoders[cat_feature] = le
        
        # Variable cible (churn basé sur probabilité)
        y = (data['churn_probability'] > 0.5).astype(int)
        
        # Division train/test
        X_train, X_test, y_train, y_test
