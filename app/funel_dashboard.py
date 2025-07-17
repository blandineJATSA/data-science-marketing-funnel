import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="🚀 Marketing Funnel Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .insight-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">🚀 Marketing Funnel Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Expert Data Science Marketing Analysis</p>', unsafe_allow_html=True)

# Fonction de génération des données (version complète)
@st.cache_data
def load_complete_data():
    """Génération des données complètes avec toutes les analyses"""
    np.random.seed(42)
    
    # Données de base
    n_users = 5000
    user_journey = pd.DataFrame({
        'user_id': range(1, n_users + 1),
        'device': np.random.choice(['desktop', 'mobile', 'tablet'], n_users, p=[0.6, 0.35, 0.05]),
        'traffic_source': np.random.choice(['organic', 'paid_search', 'social', 'email', 'direct'], n_users, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'date': pd.date_range('2015-01-01', periods=n_users, freq='H'),
        'hour': np.random.randint(0, 24, n_users),
        'reached_basic_info': np.random.choice([0, 1], n_users, p=[0.15, 0.85]),
        'reached_email': np.random.choice([0, 1], n_users, p=[0.25, 0.75]),
        'reached_job': np.random.choice([0, 1], n_users, p=[0.35, 0.65]),
        'reached_submit': np.random.choice([0, 1], n_users, p=[0.45, 0.55]),
        'reached_confirmation': np.random.choice([0, 1], n_users, p=[0.6, 0.4]),
    })
    
    # Nettoyage des données
    mask = (user_journey['reached_basic_info'] == 1) & (user_journey['reached_email'] == 1)
    clean_data = user_journey[mask].copy()
    
    # Métriques avancées
    clean_data['recency'] = (pd.to_datetime('2015-05-01') - clean_data['date']).dt.days
    clean_data['frequency'] = np.random.poisson(2, len(clean_data)) + 1
    clean_data['monetary_value'] = np.where(
        clean_data['reached_confirmation'] == 1,
        np.random.normal(120, 40, len(clean_data)),
        np.random.normal(80, 30, len(clean_data))
    )
    clean_data['monetary_value'] = np.clip(clean_data['monetary_value'], 10, 500)
    
    # Scoring RFM simplifié
    clean_data['R_Score'] = pd.qcut(clean_data['recency'], 3, labels=[3,2,1], duplicates='drop').astype(int)
    clean_data['F_Score'] = pd.qcut(clean_data['frequency'], 3, labels=[1,2,3], duplicates='drop').astype(int)
    clean_data['M_Score'] = pd.qcut(clean_data['monetary_value'], 3, labels=[1,2,3], duplicates='drop').astype(int)
    
    # Segmentation RFM
    def classify_rfm(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        if r >= 2 and f >= 2 and m >= 2:
            return 'Champions'
        elif r >= 2 and f >= 2:
            return 'Loyal Customers'
        elif r >= 2 and f <= 1:
            return 'New Customers'
        elif r <= 1 and f >= 2:
            return 'At Risk'
        elif r <= 1 and f <= 1:
            return 'Lost Customers'
        elif m >= 3:
            return 'Big Spenders'
        else:
            return 'Regular Customers'
    
    clean_data['RFM_Segment'] = clean_data.apply(classify_rfm, axis=1)
    
    # CLV et probabilité de churn
    clean_data['clv'] = clean_data['monetary_value'] * clean_data['frequency'] * np.random.uniform(0.8, 1.2, len(clean_data))
    clean_data['churn_probability'] = np.where(
        clean_data['RFM_Segment'].isin(['At Risk', 'Lost Customers']),
        np.random.uniform(0.6, 0.9, len(clean_data)),
        np.random.uniform(0.1, 0.4, len(clean_data))
    )
    
    # Attribution multi-touch
    clean_data['first_touch'] = clean_data['traffic_source']
    clean_data['last_touch'] = np.random.choice(['organic', 'paid_search', 'social', 'email', 'direct'], len(clean_data))
    
    return clean_data

# Chargement des données
with st.spinner('🔄 Chargement des données...'):
    data = load_complete_data()

# Sidebar pour les filtres
st.sidebar.header('🎛️ Filtres et Contrôles')

# Sélection de la période
date_range = st.sidebar.date_input(
    "📅 Période d'analyse",
    value=(data['date'].min().date(), data['date'].max().date()),
    min_value=data['date'].min().date(),
    max_value=data['date'].max().date()
)

# Filtres multiples
selected_devices = st.sidebar.multiselect(
    '📱 Dispositifs',
    options=data['device'].unique(),
    default=data['device'].unique()
)

selected_sources = st.sidebar.multiselect(
    '🔗 Sources de trafic',
    options=data['traffic_source'].unique(),
    default=data['traffic_source'].unique()
)

selected_segments = st.sidebar.multiselect(
    '🎯 Segments RFM',
    options=data['RFM_Segment'].unique(),
    default=data['RFM_Segment'].unique()
)

# Filtrage des données
filtered_data = data[
    (data['device'].isin(selected_devices)) &
    (data['traffic_source'].isin(selected_sources)) &
    (data['RFM_Segment'].isin(selected_segments))
]

# Métriques principales
st.header('📊 Métriques Clés')

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_users = len(filtered_data)
    st.metric("👥 Total Utilisateurs", f"{total_users:,}")

with col2:
    conversion_rate = (filtered_data['reached_confirmation'].sum() / len(filtered_data) * 100)
    st.metric("📈 Taux de Conversion", f"{conversion_rate:.1f}%")

with col3:
    avg_clv = filtered_data['clv'].mean()
    st.metric("💰 CLV Moyen", f"${avg_clv:.0f}")

with col4:
    total_revenue = filtered_data['monetary_value'].sum()
    st.metric("💵 Revenus Total", f"${total_revenue:,.0f}")

with col5:
    avg_churn = filtered_data['churn_probability'].mean() * 100
    st.metric("⚠️ Risque Churn", f"{avg_churn:.1f}%")

# Tabs pour organiser les analyses
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Analyse Funnel", 
    "🎯 Segmentation RFM", 
    "📊 Attribution Multi-Touch", 
    "🔮 Modèles Prédictifs", 
    "📈 Insights Avancés"
])

with tab1:
    st.header('🔍 Analyse du Funnel de Conversion')
    
    # Calcul des métriques du funnel
    funnel_metrics = {
        'Visiteurs': len(filtered_data),
        'Basic Info': filtered_data['reached_basic_info'].sum(),
        'Email': filtered_data['reached_email'].sum(),
        'Job Info': filtered_data['reached_job'].sum(),
        'Submit': filtered_data['reached_submit'].sum(),
        'Confirmation': filtered_data['reached_confirmation'].sum()
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique funnel
        fig_funnel = go.Figure(go.Funnel(
            y=list(funnel_metrics.keys()),
            x=list(funnel_metrics.values()),
            textinfo="value+percent initial+percent previous",
            marker=dict(color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
        ))
        fig_funnel.update_layout(title="🔍 Tunnel de Conversion", height=500)
        st.plotly_chart(fig_funnel, use_container_width=True)
    
    with col2:
        # Taux de conversion par étapes
        conversion_rates = []
        steps = list(funnel_metrics.keys())
        values = list(funnel_metrics.values())
        
        for i in range(1, len(values)):
            if values[i-1] > 0:
                rate = (values[i] / values[i-1]) * 100
                conversion_rates.append(rate)
            else:
                conversion_rates.append(0)
        
        fig_rates = px.bar(
            x=steps[1:],
            y=conversion_rates,
            title="📈 Taux de Conversion par Étape",
            labels={'x': 'Étapes', 'y': 'Taux de Conversion (%)'},
            color=conversion_rates,
            color_continuous_scale='RdYlBu_r'
        )
        fig_rates.update_layout(height=500)
        st.plotly_chart(fig_rates, use_container_width=True)
    
    # Analyse par source de trafic
    st.subheader('🔗 Performance par Source de Trafic')
    
    traffic_analysis = filtered_data.groupby('traffic_source').agg({
        'user_id': 'count',
        'reached_confirmation': 'sum',
        'monetary_value': 'mean',
        'clv': 'mean'
    }).reset_index()
    
    traffic_analysis['conversion_rate'] = (traffic_analysis['reached_confirmation'] / traffic_analysis['user_id'] * 100).round(2)
    traffic_analysis['total_revenue'] = traffic_analysis['user_id'] * traffic_analysis['monetary_value']
    
    fig_traffic = px.scatter(
        traffic_analysis,
        x='conversion_rate',
        y='clv',
        size='user_id',
        color='traffic_source',
        hover_data=['total_revenue'],
        title="🎯 Performance Sources: Conversion vs CLV",
        labels={'conversion_rate': 'Taux de Conversion (%)', 'clv': 'CLV ( $ )'}
    )
    st.plotly_chart(fig_traffic, use_container_width=True)

with tab2:
    st.header('🎯 Analyse de Segmentation RFM')
    
    # Analyse RFM
    rfm_analysis = filtered_data.groupby('RFM_Segment').agg({
        'user_id': 'count',
        'reached_confirmation': 'sum',
        'monetary_value': 'mean',
        'clv': 'mean',
        'churn_probability': 'mean',
        'recency': 'mean',
        'frequency': 'mean'
    }).round(2)
    
    rfm_analysis['conversion_rate'] = (rfm_analysis['reached_confirmation'] / rfm_analysis['user_id'] * 100).round(2)
    rfm_analysis['total_value'] = (rfm_analysis['user_id'] * rfm_analysis['monetary_value']).round(0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution des segments
        fig_segments = px.pie(
            values=rfm_analysis['user_id'],
            names=rfm_analysis.index,
            title="🥧 Distribution des Segments RFM"
        )
        st.plotly_chart(fig_segments, use_container_width=True)
    
    with col2:
        # Matrice RFM
        fig_rfm_matrix = px.scatter(
            rfm_analysis.reset_index(),
            x='conversion_rate',
            y='clv',
            size='user_id',
            color='RFM_Segment',
            title="📊 Matrice RFM: Conversion vs CLV",
            labels={'conversion_rate': 'Taux de Conversion (%)', 'clv': 'CLV ( $ )'}
        )
        st.plotly_chart(fig_rfm_matrix, use_container_width=True)
    
    # Tableau détaillé
    st.subheader('📋 Analyse Détaillée par Segment')
    
    # Formatage du tableau
    rfm_display = rfm_analysis.copy()
    rfm_display['churn_probability'] = (rfm_display['churn_probability'] * 100).round(1)
    
    st.dataframe(
        rfm_display.style.format({
            'monetary_value': '${:.0f}',
            'clv': '${:.0f}',
            'total_value': '${:.0f}',
            'conversion_rate': '{:.1f}%',
            'churn_probability': '{:.1f}%'
        }),
        use_container_width=True
    )
    
    # Recommandations stratégiques
    st.subheader('🎯 Recommandations Stratégiques')
    
    strategies = {
        'Champions': {'color': '#2ca02c', 'strategy': '🏆 Récompenser, programmes VIP, ambassadeurs'},
        'Loyal Customers': {'color': '#1f77b4', 'strategy': '💎 Up-sell, cross-sell, programme fidélité'},
        'New Customers': {'color': '#ff7f0e', 'strategy': '🌟 Onboarding, recommandations, support'},
        'At Risk': {'color': '#d62728', 'strategy': '⚠️ Campagnes réactivation, offres spéciales'},
        'Lost Customers': {'color': '#9467bd', 'strategy': '💔 Win-back campaigns, sondages abandon'},
        'Big Spenders': {'color': '#8c564b', 'strategy': '💰 Produits premium, service personnalisé'},
        'Regular Customers': {'color': '#e377c2', 'strategy': '📈 Activation frequency, gamification'}
    }
    
    for segment in rfm_analysis.index:
        if segment in strategies:
            with st.expander(f"{segment} ({rfm_analysis.loc[segment, 'user_id']} users)"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Conv. Rate", f"{rfm_analysis.loc[segment, 'conversion_rate']:.1f}%")
                with col2:
                    st.metric("CLV", f"${rfm_analysis.loc[segment, 'clv']:.0f}")
                with col3:
                    st.metric("Churn Risk", f"{rfm_analysis.loc[segment, 'churn_probability']:.1f}%")
                
                st.info(strategies[segment]['strategy'])

with tab3:
    st.header('📊 Attribution Multi-Touch')
    
    # Analyse d'attribution
    attribution_analysis = filtered_data.groupby(['first_touch', 'last_touch']).agg({
        'user_id': 'count',
        'reached_confirmation': 'sum',
        'monetary_value': 'mean'
    }).reset_index()
    
    attribution_analysis['conversion_rate'] = (attribution_analysis['reached_confirmation'] / attribution_analysis['user_id'] * 100).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # First touch attribution
        first_touch = filtered_data.groupby('first_touch').agg({
            'user_id': 'count',
            'reached_confirmation': 'sum',
            'monetary_value': 'sum'
        }).reset_index()
        
        fig_first = px.bar(
            first_touch,
            x='first_touch',
            y='monetary_value',
            title="🎯 Attribution Premier Touch",
            labels={'first_touch': 'Canal', 'monetary_value': 'Revenus ( $ )'}
        )
        st.plotly_chart(fig_first, use_container_width=True)
    
    with col2:
        # Last touch attribution
        last_touch = filtered_data.groupby('last_touch').agg({
            'user_id': 'count',
            'reached_confirmation': 'sum',
            'monetary_value': 'sum'
        }).reset_index()
        
        fig_last = px.bar(
            last_touch,
            x='last_touch',
            y='monetary_value',
            title="🎯 Attribution Dernier Touch",
            labels={'last_touch': 'Canal', 'monetary_value': 'Revenus ( $ )'}
        )
        st.plotly_chart(fig_last, use_container_width=True)
    
    # Matrice d'attribution
    st.subheader('🔄 Matrice d\'Attribution Cross-Canal')
    
    # Créer la matrice pivot
    attribution_matrix = filtered_data.pivot_table(
        values='monetary_value',
        index='first_touch',
        columns='last_touch',
        aggfunc='sum',
        fill_value=0
    )
    
    fig_matrix = px.imshow(
        attribution_matrix,
        labels=dict(x="Dernier Touch", y="Premier Touch", color="Revenus"),
        title="🔄 Matrice d'Attribution Cross-Canal",
        color_continuous_scale='RdYlBu_r'
    )
    st.plotly_chart(fig_matrix, use_container_width=True)

with tab4:
    st.header('🔮 Modèles Prédictifs')
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Prédiction du churn
    st.subheader('⚠️ Modèle de Prédiction du Churn')
    
    # Préparation des données pour le modèle
    features = ['recency', 'frequency', 'monetary_value', 'R_Score', 'F_Score', 'M_Score']
    X = filtered_data[features]
    y = (filtered_data['churn_probability'] > 0.5).astype(int)
    
    # Encodage des variables catégorielles
    device_encoded = pd.get_dummies(filtered_data['device'], prefix='device')
    source_encoded = pd.get_dummies(filtered_data['traffic_source'], prefix='source')
    
    X_enhanced = pd.concat([X, device_encoded, source_encoded], axis=1)
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(X_enhanced, y, test_size=0.3, random_state=42)
    
    # Entraînement du modèle
    with st.spinner('🤖 Entraînement du modèle...'):
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = rf_model.predict(X_test)
    accuracy = rf_model.score(X_test, y_test)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🎯 Précision du Modèle", f"{accuracy:.2%}")
        
        # Importance des features
        feature_importance = pd.DataFrame({
            'feature': X_enhanced.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        fig_importance = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title="🔍 Importance des Variables",
            labels={'importance': 'Importance', 'feature': 'Variables'}
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col2:
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Prédiction", y="Réalité", color="Nombre"),
            title="📊 Matrice de Confusion",
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    # Prédiction CLV
    st.subheader('💰 Prédiction de la Customer Lifetime Value')
    
    # Statistiques CLV par segment
    clv_stats = filtered_data.groupby('RFM_Segment')['clv'].agg(['mean', 'std', 'min', 'max']).round(2)
    
    fig_clv = px.box(
        filtered_data,
        x='RFM_Segment',
        y='clv',
        title="📊 Distribution CLV par Segment",
        labels={'clv': 'CLV ( $ )', 'RFM_Segment': 'Segment RFM'}
    )
    st.plotly_chart(fig_clv, use_container_width=True)
    
    # Recommandations basées sur les prédictions
    st.subheader('🎯 Recommandations Basées sur les Prédictions')
    
    # Identifier les clients à risque
    high_risk_customers = filtered_data[filtered_data['churn_probability'] > 0.7]
    high_value_customers = filtered_data[filtered_data['clv'] > filtered_data['clv'].quantile(0.8)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.error(f"⚠️ **{len(high_risk_customers)} clients à haut risque de churn**")
        st.write("Actions recommandées:")
        st.write("- Campagnes de rétention personnalisées")
        st.write("- Offres spéciales et réductions")
        st.write("- Contact direct par l'équipe commerciale")
    
    with col2:
        st.success(f"💎 **{len(high_value_customers)} clients à haute valeur**")
        st.write("Actions recommandées:")
        st.write("- Programmes VIP et récompenses")
        st.write("- Up-selling et cross-selling")
        st.write("- Service client premium")

with tab5:
    st.header('📈 Insights Avancés')
    
    # Analyse temporelle
    st.subheader('⏰ Analyse Temporelle')
    
    # Conversion par heure
    hourly_analysis = filtered_data.groupby('hour').agg({
        'user_id': 'count',
        'reached_confirmation': 'sum'
    }).reset_index()
    hourly_analysis['conversion_rate'] = (hourly_analysis['reached_confirmation'] / hourly_analysis['user_id'] * 100).round(2)
    
    fig_hourly = px.line(
        hourly_analysis,
        x='hour',
        y='conversion_rate',
        title="📊 Taux de Conversion par Heure",
        labels={'hour': 'Heure', 'conversion_rate': 'Taux de Conversion (%)'}
    )
    fig_hourly.update_layout(xaxis=dict(tickmode='linear'))
    st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Analyse cohort simulation
    st.subheader('👥 Analyse de Cohorte (Simulation)')
    
    # Simuler des données de cohorte
    cohort_data = []
    for month in range(1, 13):
        retention_rate = max(0.1, 0.9 - (month-1) * 0.07)  # Décroissance réaliste
        cohort_data.append({
            'month': month,
            'retention_rate': retention_rate * 100,
            'users_retained': int(1000 * retention_rate)
        })
    
    cohort_df = pd.DataFrame(cohort_data)
    
    fig_cohort = px.bar(
        cohort_df,
        x='month',
        y='retention_rate',
        title="📊 Taux de Rétention par Cohorte Mensuelle",
        labels={'month': 'Mois', 'retention_rate': 'Taux de Rétention (%)'}
    )
    st.plotly_chart(fig_cohort, use_container_width=True)
    
    # Recommandations personnalisées
    st.subheader('🎯 Système de Recommandations Personnalisées')
    
    # Simuler des recommandations
    recommendation_impact = {
        'Produits Recommandés': np.random.randint(50, 200, 10),
        'Uplift Estimé (%)': np.random.uniform(15, 45, 10),
        'Revenus Additionnels ( $ )': np.random.randint(5000, 25000, 10)
    }
    
    rec_df = pd.DataFrame(recommendation_impact)
    rec_df['Campagne'] = [f'Campagne {i+1}' for i in range(10)]
    
    fig_rec = px.scatter(
        rec_df,
        x='Uplift Estimé (%)',
        y='Revenus Additionnels ( $ )',
        size='Produits Recommandés',
        hover_data=['Campagne'],
        title="🎯 Impact des Recommandations Personnalisées",
        labels={'Uplift Estimé (%)': 'Uplift (%)', 'Revenus Additionnels ( $ )': 'Revenus ($)'}
    )
    st.plotly_chart(fig_rec, use_container_width=True)
    
    # Insights finaux
    st.subheader('💡 Insights Clés et Recommandations')
    
    # Calcul des KPIs globaux
    total_revenue = filtered_data['monetary_value'].sum()
    avg_clv = filtered_data['clv'].mean()
    conversion_rate = filtered_data['reached_confirmation'].mean() * 100
    
    insights = [
        f"🎯 **Conversion Rate**: {conversion_rate:.1f}% - {'Excellent' if conversion_rate > 25 else 'À améliorer'}",
        f"💰 **CLV Moyen**: ${avg_clv:.0f} - {'Très bon' if avg_clv > 200 else 'Potentiel d\'amélioration'}",
        f"🔍 **Meilleur Segment**: {rfm_analysis['conversion_rate'].idxmax()} avec {rfm_analysis['conversion_rate'].max():.1f}% de conversion",
        f"⚠️ **Segment à Risque**: {rfm_analysis['churn_probability'].idxmax()} avec {rfm_analysis['churn_probability'].max()*100:.1f}% de risque churn"
    ]
    
    for insight in insights:
        st.info(insight)
    
    # Actions recommandées
    st.subheader('🚀 Plan d\'Action Recommandé')
    
    actions = [
        "📈 **Court terme**: Optimiser les heures de pic de conversion (8h-18h)",
        "🎯 **Moyen terme**: Campagnes de rétention pour les segments 'At Risk'",
        "💎 **Long terme**: Programme VIP pour les 'Champions' et 'Big Spenders"
    ]