"""
Package des composants pour le Dashboard Marketing Analytics
"""

from .data_processing import (
    load_and_clean_data,
    create_rfm_segments,
    calculate_clv,
    prepare_funnel_data
)

from .visualizations import (
    create_funnel_chart,
    create_rfm_heatmap,
    create_conversion_timeline,
    create_attribution_sankey
)

from .analytics import (
    analyze_customer_segments,
    predict_churn,
    calculate_attribution,
    generate_recommendations
)

__version__ = "1.0.0"
__author__ = "Data Science Marketing Team"

# Configuration globale
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17a2b8'
}

SEGMENT_COLORS = {
    'Champions': '#2ca02c',
    'Loyal Customers': '#1f77b4',
    'New Customers': '#ff7f0e',
    'At Risk': '#d62728',
    'Lost Customers': '#9467bd',
    'Big Spenders': '#8c564b',
    'Regular Customers': '#e377c2'
}

# Métadonnées
FUNNEL_STEPS = [
    'Visiteurs',
    'Basic Info',
    'Email',
    'Job Info', 
    'Submit',
    'Confirmation'
]

RFM_SEGMENTS = [
    'Champions',
    'Loyal Customers',
    'New Customers',
    'At Risk',
    'Lost Customers',
    'Big Spenders',
    'Regular Customers'
]
