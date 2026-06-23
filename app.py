import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration for a premium look
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
    /* Global styles */
    .main {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #e2e8f0;
        font-weight: 600;
    }
    .header-text {
        font-size: 2.5rem;
        background: -webkit-linear-gradient(45deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    /* Sub-header */
    .sub-header {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Cards for recommendations */
    .stCard {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #334155;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 15px;
    }
    .stCard:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border-color: #3b82f6;
    }
    .card-title {
        color: #f8fafc;
        font-size: 1.1rem;
        font-weight: 500;
        margin: 0;
    }
    .card-icon {
        color: #8b5cf6;
        font-size: 1.5rem;
        margin-right: 10px;
        vertical-align: middle;
    }
    
    /* Cluster display */
    .cluster-box {
        text-align: center;
        padding: 30px;
        border-radius: 12px;
        margin-top: 20px;
    }
    .cluster-High-Value {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    .cluster-Regular {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
    }
    .cluster-Occasional {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    .cluster-At-Risk {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    .cluster-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .cluster-desc {
        font-size: 1rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load('rfm_scaler.pkl')
        kmeans = joblib.load('rfm_kmeans.pkl')
        label_map = joblib.load('cluster_label_map.pkl')
        recommendations = joblib.load('product_recommendations.pkl')
        product_list = joblib.load('product_list.pkl')
        return scaler, kmeans, label_map, recommendations, product_list
    except Exception as e:
        return None, None, None, None, None

scaler, kmeans, label_map, recommendations, product_list = load_models()

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3081/3081840.png", width=100)
st.sidebar.title("Shopper Spectrum")
st.sidebar.markdown("Navigate through the application modules:")
page = st.sidebar.radio("Select Module", ["Product Recommendation", "Customer Segmentation"])

st.sidebar.markdown("---")
st.sidebar.info("This application uses Machine Learning to provide product recommendations based on collaborative filtering and segments customers using RFM analysis and KMeans clustering.")

if scaler is None:
    st.error("Models not found! Please ensure you have run the analysis script to generate the models.")
    st.stop()

if page == "Product Recommendation":
    st.markdown("<h1 class='header-text'>🛍️ Product Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Discover products similar to what you're looking for based on collaborative filtering.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Search Product")
        selected_product = st.selectbox("Select or type a product name:", [""] + sorted(product_list))
        
        if st.button("Get Recommendations", type="primary", use_container_width=True):
            if selected_product:
                with st.spinner("Finding best matches..."):
                    recs = recommendations.get(selected_product, [])
                    if recs:
                        st.markdown("### Top 5 Recommended Products")
                        for i, prod in enumerate(recs):
                            st.markdown(f"""
                            <div class="stCard">
                                <span class="card-icon">✧</span>
                                <span class="card-title">{prod}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("No recommendations found for this product.")
            else:
                st.warning("Please select a product first.")

elif page == "Customer Segmentation":
    st.markdown("<h1 class='header-text'>👥 Customer Segmentation Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Input customer RFM metrics to predict their segment in real-time.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Enter Customer Metrics")
        with st.form("segmentation_form"):
            recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30, step=1)
            frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=5, step=1)
            monetary = st.number_input("Monetary (total spend in £)", min_value=0.0, value=250.0, step=10.0)
            
            submit_button = st.form_submit_button(label="Predict Cluster", use_container_width=True)
    
    with col2:
        if submit_button:
            st.markdown("### Prediction Result")
            # Create dataframe for prediction
            input_data = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
            
            # Scale
            scaled_data = scaler.transform(input_data)
            
            # Predict
            cluster_id = kmeans.predict(scaled_data)[0]
            cluster_label = label_map[cluster_id]
            
            # Display logic
            desc_map = {
                "High-Value": "Regular, frequent, recent, and big spenders. Offer them premium services and loyalty programs.",
                "Regular": "Steady purchasers but not premium. Send them personalized recommendations to increase their basket size.",
                "Occasional": "Rare, occasional purchases. Engage them with discounts and promotional campaigns.",
                "At-Risk": "Haven't purchased in a long time. Need immediate reactivation campaigns to bring them back."
            }
            
            # CSS class name mapping
            css_class = f"cluster-{cluster_label}"
            
            st.markdown(f"""
            <div class="cluster-box {css_class}">
                <div class="cluster-title">🎯 {cluster_label} Customer</div>
                <div class="cluster-desc">{desc_map.get(cluster_label, "")}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show the raw scaled values for context
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("View Technical Details"):
                st.write(f"Assigned Cluster ID: {cluster_id}")
                st.write(f"Scaled Recency: {scaled_data[0][0]:.3f}")
                st.write(f"Scaled Frequency: {scaled_data[0][1]:.3f}")
                st.write(f"Scaled Monetary: {scaled_data[0][2]:.3f}")
