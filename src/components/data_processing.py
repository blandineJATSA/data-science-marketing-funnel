"""
Module de traitement et nettoyage des données marketing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from typing import Dict, List, Tuple, Optional

class DataProcessor:
    """Classe principale pour le traitement des données marketing"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_synthetic_data(self, n_users: int = 5000) -> pd.DataFrame:
        """
        Génère des données synthétiques pour le funnel marketing
        
        Args:
            n_users: Nombre d'utilisateurs à générer
            
        Returns:
            DataFrame avec les données utilisateur
        """
        # Génération des données de base
        data = pd.DataFrame({
            'user_id': range(1, n_users + 1),
            'device': np.random.choice(['desktop', 'mobile', 'tablet'], n_users, p=[0.6, 0.35, 0.05]),
            'traffic_source': np.random.choice(['organic', 'paid_search', 'social', 'email', 'direct'], 
                                             n_users, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
            'date': pd.date_range('2015-01-01', periods=n_users, freq='H'),
            'hour': np.random.randint(0, 24, n_users),
            'country': np.random.choice(['US', 'UK', 'FR', 'DE', 'CA'], n_users),
            'session_duration': np.random.exponential(300, n_users),  # en secondes
            'page_views': np.random.poisson(5, n_users) + 1,
        })
        
        # Création du funnel avec dépendances réalistes
        data['reached_basic_info'] = self._simulate_funnel_step(n_users, 0.85)
        data['reached_email'] = self._simulate_dependent_step(
            data['reached_basic_info'], 0.88
        )
        data['reached_job'] = self._simulate_dependent_step(
            data['reached_email'], 0.75
        )
        data['reached_submit'] = self._simulate_dependent_step(
            data['reached_job'], 0.65
        )
        data['reached_confirmation'] = self._simulate_dependent_step(
            data['reached_submit'], 0.80
        )
        
        return data
    
    def _simulate_funnel_step(self, n_users: int, base_rate: float) -> np.ndarray:
        """Simule une étape du funnel"""
        return np.random.choice([0, 1], n_users, p=[1-base_rate, base_rate])
    
    def _simulate_dependent_step(self, previous_step: pd.Series, conditional_rate: float) -> np.ndarray:
        """Simule une étape dépendante de la précédente"""
        result = np.zeros(len(previous_step))
        for i, prev_value in enumerate(previous_step):
            if prev_value == 1:
                result[i] = np.random.choice([0, 1], p=[1-conditional_rate, conditional_rate])
        return result.astype(int)
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les données en supprimant les incohérences
        
        Args:
            data: DataFrame avec les données brutes
            
        Returns:
            DataFrame nettoyé
        """
        # Filtrer les utilisateurs qui ont au moins atteint basic_info et email
        mask = (data['reached_basic_info'] == 1) & (data['reached_email'] == 1)
        cleaned_data = data[mask].copy()
        
        # Supprimer les doublons
        cleaned_data = cleaned_data.drop_duplicates(subset=['user_id'])
        
        # Gérer les valeurs aberrantes
        cleaned_data['session_duration'] = np.clip(cleaned_data['session_duration'], 10, 3600)
        cleaned_data['page_views'] = np.clip(cleaned_data['page_views'], 1, 50)
        
        return cleaned_data.reset_index(drop=True)
    
    def create_rfm_metrics(self, data: pd.DataFrame, reference_date: str = '2015-05-01') -> pd.DataFrame:
        """
        Crée les métriques RFM (Recency, Frequency, Monetary)
        
        Args:
            data: DataFrame avec les données nettoyées
            reference_date: Date de référence pour calculer la récence
            
        Returns:
            DataFrame avec les métriques RFM
        """
        data = data.copy()
        
        # Recency: Jours depuis la dernière visite
        data['recency'] = (pd.to_datetime(reference_date) - pd.to_datetime(data['date'])).dt.days
        
        # Frequency: Simulation basée sur les pages vues et durée de session
        data['frequency'] = np.random.poisson(
            np.clip(data['page_views'] / 2, 1, 10)
        ) + 1
        
        # Monetary: Valeur estimée basée sur le comportement
        conversion_multiplier = np.where(data['reached_confirmation'] == 1, 1.5, 1.0)
        device_multiplier = np.where(data['device'] == 'desktop', 1.2, 
                                   np.where(data['device'] == 'mobile', 1.0, 0.9))
        
        data['monetary_value'] = (
            np.random.normal(100, 30, len(data)) * 
            conversion_multiplier * 
            device_multiplier
        )
        data['monetary_value'] = np.clip(data['monetary_value'], 10, 500)
        
        return data
    
    def create_rfm_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Crée les scores RFM et les segments
        
        Args:
            data: DataFrame avec les métriques RFM
            
        Returns:
            DataFrame avec les scores et segments RFM
        """
        data = data.copy()
        
        # Fonction robuste pour créer les scores
        def safe_qcut(series: pd.Series, q: int = 5, ascending: bool = True) -> pd.Series:
            """Création sécurisée des quantiles"""
            try:
                labels = list(range(1, q+1)) if ascending else list(range(q, 0, -1))
                return pd.qcut(series, q, labels=labels, duplicates='drop').astype(int)
            except ValueError:
                # Fallback si pas assez de valeurs uniques
                n_unique = series.nunique()
                if n_unique < q:
                    q = n_unique
                    labels = list(range(1, q+1)) if ascending else list(range(q, 0, -1))
                return pd.qcut(series, q, labels=labels, duplicates='drop').astype(int)
        
        # Création des scores
        data['R_Score'] = safe_qcut(data['recency'], ascending=False)
        data['F_Score'] = safe_qcut(data['frequency'], ascending=True)
        data['M_Score'] = safe_qcut(data['monetary_value'], ascending=True)
        
        # Score combiné
        data['RFM_Score'] = (
            data['R_Score'].astype(str) + 
            data['F_Score'].astype(str) + 
            data['M_Score'].astype(str)
        )
        
        # Segmentation
        data['RFM_Segment'] = data.apply(self._classify_rfm_segment, axis=1)
        
        return data
    
    def _classify_rfm_segment(self, row: pd.Series) -> str:
        """Classifie un utilisateur dans un segment RFM"""
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f <= 2:
            return 'New Customers'
        elif r <= 2 and f >= 3:
            return 'At Risk'
        elif r <= 2 and f <= 2:
            return 'Lost Customers'
        elif m >= 4:
            return 'Big Spenders'
        elif r >= 3 and f <= 2:
            return 'Potential Loyalists'
        else:
            return 'Regular Customers'
    
    def calculate_advanced_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule des métriques avancées (CLV, probabilité de churn, etc.)
        
        Args:
            data: DataFrame avec les données RFM
            
        Returns:
            DataFrame avec les métriques avancées
        """
        data = data.copy()
        
        # Customer Lifetime Value
        data['clv'] = (
            data['monetary_value'] * 
            data['frequency'] * 
            np.random.uniform(0.8, 1.2, len(data))
        )
        
        # Probabilité de churn basée sur RFM
        churn_mapping = {
            'Champions': (0.05, 0.15),
            'Loyal Customers': (0.10, 0.25),
            'New Customers': (0.15, 0.35),
            'At Risk': (0.60, 0.85),
            'Lost Customers': (0.80, 0.95),
            'Big Spenders': (0.10, 0.20),
            'Potential Loyalists': (0.20, 0.40),
            'Regular Customers': (0.25, 0.45)
        }
        
        data['churn_probability'] = data['RFM_Segment'].apply(
            lambda x: np.random.uniform(*churn_mapping.get(x, (0.3, 0.5)))
        )
        
        # Score d'engagement
        data['engagement_score'] = (
            data['page_views'] * 0.3 + 
            (data['session_duration'] / 60) * 0.2 + 
            data['frequency'] * 0.5
        )
        
        # Attribution multi-touch
        data['first_touch'] = data['traffic_source']
        data['last_touch'] = np.random.choice(
            ['organic', 'paid_search', 'social', 'email', 'direct'],
            len(data)
        )
        
        return data
    
    def save_processed_data(self, data: pd.DataFrame, filepath: str) -> None:
        """
        Sauvegarde les données traitées
        
        Args:
            data: DataFrame à sauvegarder
            filepath: Chemin du fichier de sauvegarde
        """
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ Données sauvegardées dans {filepath}")
    
    def load_processed_data(self, filepath: str) -> pd.DataFrame:
        """
        Charge les données traitées
        
        Args:
            filepath: Chemin du fichier à charger
            
        Returns:
            DataFrame avec les données chargées
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print(f"✅ Données chargées depuis {filepath}")
            return data
        except FileNotFoundError:
            print(f"❌ Fichier {filepath} non trouvé")
            return None

def load_and_clean_data(use_cache: bool = True, cache_path: str = 'data/processed_data.pkl') -> pd.DataFrame:
    """
    Fonction principale pour charger et nettoyer les données
    
    Args:
        use_cache: Utiliser le cache si disponible
        cache_path: Chemin du fichier de cache
        
    Returns:
        DataFrame avec les données nettoyées et enrichies
    """
    processor = DataProcessor()
    
    # Vérifier si le cache existe
    if use_cache:
        try:
            cached_data = processor.load_processed_data(cache_path)
            if cached_data is not None:
                return cached_data
        except:
            pass
    
    # Générer de nouvelles données
    print("🔄 Génération de nouvelles données...")
    raw_data = processor.generate_synthetic_data(5000)
    
    # Nettoyer les données
    print("🧹 Nettoyage des données...")
    clean_data = processor.clean_data(raw_data)
    
    # Créer les métriques RFM
    print("📊 Création des métriques RFM...")
    rfm_data = processor.create_rfm_metrics(clean_data)
    
    # Créer les scores RFM
    print("🎯 Création des scores et segments RFM...")
    scored_data = processor.create_rfm_scores(rfm_data)
    
    # Calculer les métriques avancées
    print("⚡ Calcul des métriques avancées...")
    final_data = processor.calculate_advanced_metrics(scored_data)
    
    # Sauvegarder
    if use_cache:
        processor.save_processed_data(final_data, cache_path)
    
    return final_data

# Fonctions utilitaires
def create_rfm_segments(data: pd.DataFrame) -> Dict:
    """Crée un résumé des segments RFM"""
    return data.groupby('RFM_Segment').agg({
        'user_id': 'count',
        'reached_confirmation': 'sum',
        'monetary_value': 'mean',
        'clv': 'mean',
        'churn_probability': 'mean'
    }).to_dict()

def calculate_clv(data: pd.DataFrame) -> Dict:
    """Calcule les statistiques CLV"""
    return {
        'mean': data['clv'].mean(),
        'median': data['clv'].median(),
        'std': data['clv'].std(),
        'total': data['clv'].sum()
    }

def prepare_funnel_data(data: pd.DataFrame) -> Dict:
    """Prépare les données du funnel"""
    return {
        'Visiteurs': len(data),
        'Basic Info': data['reached_basic_info'].sum(),
        'Email': data['reached_email'].sum(),
        'Job Info': data['reached_job'].sum(),
        'Submit': data['reached_submit'].sum(),
        'Confirmation': data['reached_confirmation'].sum()
    }
