"""
Module de visualisations pour le dashboard marketing
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from . import COLORS, SEGMENT_COLORS

class MarketingVisualizer:
    """Classe pour créer les visualisations marketing"""
    
    def __init__(self):
        self.colors = COLORS
        self.segment_colors = SEGMENT_COLORS
    
    def create_funnel_chart(self, funnel_data: Dict, title: str = "🔍 Tunnel de Conversion") -> go.Figure:
        """
        Crée un graphique en entonnoir
        
        Args:
            funnel_data: Dictionnaire avec les données du funnel
            title: Titre du graphique
            
        Returns:
            Figure Plotly
        """
        fig = go.Figure(go.Funnel(
            y=list(funnel_data.keys()),
            x=list(funnel_data.values()),
            textinfo="value+percent initial+percent previous",
            marker=dict(
                color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
                line=dict(width=2, color="white")
            ),
            connector=dict(line=dict(color="rgb(63, 63, 63)"))
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            height=500,
            template="plotly_white"
        )
        
        return fig
    
    def create_conversion_rates_chart(self, funnel_data: Dict) -> go.Figure:
        """Crée un graphique des taux de conversion par étape"""
        steps = list(funnel_data.keys())
        values = list(funnel_data.values())
        
        conversion_rates = []
        for i in range(1, len(values)):
            if values[i-1] > 0:
                rate = (values[i] / values[i-1]) * 100
                conversion_rates.append(rate)
            else:
                conversion_rates.append(0)
        
        fig = px.bar(
            x=steps[1:],
            y=conversion_rates,
            title="📈 Taux de Conversion par Étape",
            labels={'x': 'Étapes', 'y': 'Taux de Conversion (%)'},
            color=conversion_rates,
            color_continuous_scale='RdYlBu_r'
        )
        
        fig.update_layout(
            showlegend=False,
            height=400,
            template="plotly_white"
        )
        
        return fig
    
    def create_rfm_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """Crée une heatmap des segments RFM"""
        # Créer une matrice RFM
        rfm_summary = data.groupby('RFM_Segment').agg({
            'user_id': 'count',
            'clv': 'mean',
            'churn_probability': 'mean'
        }).reset_index()
        
        fig = px.scatter(
            rfm_summary,
            x='clv',
            y='churn_probability',
            size='user_id',
            color='RFM_Segment',
            hover_data=['user_id'],
            title="📊 Matrice RFM: CLV vs Risque de Churn",
            labels={
                'clv': 'Customer Lifetime Value ($)',
                'churn_probability': 'Probabilité de Churn',
                'user_id': 'Nombre d\'utilisateurs'
            },
            color_discrete_map=self.segment_colors
        )
        
        fig.update_layout(
            height=500,
            template="plotly_white"
        )
        
        return fig
    
    def create_segment_distribution(self, data: pd.DataFrame) -> go.Figure:
        """Crée un graphique de distribution des segments"""
        segment_counts = data['RFM_Segment'].value_counts()
        
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="🥧 Distribution des Segments RFM",
            color=segment_counts.index,
            color_discrete_map=self.segment_colors
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            height=400,
            template="plotly_white"
        )
        
        return fig
    
    def create_attribution_sankey(self, data: pd.DataFrame) -> go.Figure:
        """Crée un diagramme Sankey pour l'attribution"""
        # Préparer les données pour Sankey
        attribution_data = data.groupby(['first_touch', 'last_touch']).size().reset_index(name='count')
        
        # Créer les labels uniques
        sources = data['first_touch'].unique()
        targets = data['last_touch'].unique()
        all_labels = list(sources) + [f"{t}_end" for t in targets]
        
        # Créer les liens
        source_indices = []
        target_indices = []
        values = []
        
        for _, row in attribution_data.iterrows():
            source_idx = list(sources).index(row['first_touch'])
            target_idx = len(sources) + list(targets).index(row['last_touch'])
            
            source_indices.append(source_idx)
            target_indices.append(target_idx)
            values.append(row['count'])
        
        fig = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_labels,
                color="blue"
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values
            )
        ))
        
        fig.update_layout(
            title="🔄 Attribution Multi-Touch",
            height=400,
            template="plotly_white"
        )
        
        return fig
    
    def create_conversion_timeline(self, data: pd.DataFrame) -> go.Figure:
        """Crée une timeline des conversions"""
        # Analyser les conversions par jour
        daily_conversions = data.groupby(data['date'].dt.date).agg({
            'user_id': 'count',
            'reached_confirmation': 'sum'
        }).reset_index()
        
        daily_conversions['conversion_rate'] = (
            daily_conversions['reached_confirmation'] / 
            daily_conversions['user_id'] * 100
        )
        
        fig = go.Figure()
        
        # Ajouter les utilisateurs totaux
        fig.add_trace(go.Scatter(
            x=daily_conversions['date'],
            y=daily_conversions['user_id'],
            mode='lines+markers',
            name='Total Users',
            line=dict(color='blue'),
            yaxis='y'
        ))
        
        # Ajouter le taux de conversion
        fig.add_trace(go.Scatter(
            x=daily_conversions['date'],
            y=daily_conversions['conversion_rate'],
            mode='lines+markers',
            name='Conversion Rate (%)',
            line=dict(color='red'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="📅 Évolution des Conversions dans le Temps",
            xaxis_title="Date",
            yaxis=dict(
                title="Nombre d'utilisateurs",
                side="left"
            ),
            yaxis2=dict(
                title="Taux de conversion (%)",
                side="right",
                overlaying="y"
            ),
            height=400,
            template="plotly_white"
        )
        
        return fig
    
    def create_device_performance_chart(self, data: pd.DataFrame) -> go.Figure:
        """Crée un graphique de performance par device"""
        device_analysis = data.groupby('device').agg({
            'user_id': 'count',
            'reached_confirmation': 'sum',
            'clv': 'mean'
        }).reset_index()
        
        device_analysis['conversion_rate'] = (
            device_analysis['reached_confirmation'] / 
            device_analysis['user_id'] * 100
        )
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Conversion Rate', 'CLV Moyen'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Graphique 1: Taux de conversion
        fig.add_trace(
            go.Bar(
                x=device_analysis['device'],
                y=device_analysis['conversion_rate'],
                name='Conversion Rate',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Graphique 2: CLV
        fig.add_trace(
            go.Bar(
                x=device_analysis['device'],
                y=device_analysis['clv'],
                name='CLV',
                marker_color='lightgreen'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="📱 Performance par Device",
            height=400,
            template="plotly_white",
            showlegend=False
        )
        
        return fig

# Fonctions utilitaires
def create_funnel_chart(funnel_data: Dict, title: str = "🔍 Tunnel de Conversion") -> go.Figure:
    """Fonction wrapper pour créer un graphique funnel"""
    visualizer = MarketingVisualizer()
    return visualizer.create_funnel_chart(funnel_data, title)

def create_rfm_heatmap(data: pd.DataFrame) -> go.Figure:
    """Fonction wrapper pour créer une heatmap RFM"""
    visualizer = MarketingVisualizer()
    return visualizer.create_rfm_heatmap(data)

def create_conversion_timeline(data: pd.DataFrame) -> go.Figure:
    """Fonction wrapper pour créer une timeline"""
    visualizer = MarketingVisualizer()
    return visualizer.create_conversion_timeline(data)

def create_attribution_sankey(data: pd.DataFrame) -> go.Figure:
    """Fonction wrapper pour créer un diagramme Sankey"""
    visualizer = MarketingVisualizer()
    return visualizer.create_attribution_sankey(data)
