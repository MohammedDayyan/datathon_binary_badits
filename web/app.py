"""
Image Dehazing Web Dashboard
Interactive web application for image dehazing using multiple models
"""

import streamlit as st
import sys
import os
import time
import numpy as np
from PIL import Image
import torch
import cv2
import io
import base64

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.inference import DehazeInferencePipeline
from src.models import DehazeModelManager

# Page configuration
st.set_page_config(
    page_title="Image Dehazing Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .model-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    """Load the dehazing pipeline (cached for performance)"""
    try:
        pipeline = DehazeInferencePipeline()
        pipeline.initialize_models()
        return pipeline
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        return None

def display_model_info(pipeline):
    """Display information about available models"""
    model_info = pipeline.get_model_info()
    
    st.markdown("### 🤖 Available Models")
    
    for model_name in model_info["available_models"]:
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                model_descriptions = {
                    "aodnet": "All-in-One Dehazing Network - Fast and efficient",
                    "dehazenet": "DehazeNet - End-to-end haze removal",
                    "msbdn": "Multi-Scale Boosted Dehazing Network - High quality"
                }
                
                st.markdown(f"""
                <div class="model-card">
                    <strong>{model_name.upper()}</strong><br>
                    {model_descriptions.get(model_name, "Advanced dehazing model")}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button(f"Select", key=f"select_{model_name}"):
                    st.session_state.selected_model = model_name

def process_image(pipeline, image, model_name):
    """Process a single image with the selected model"""
    if image is None:
        return None, None
        
    # Save uploaded image temporarily
    temp_input = "temp_input.jpg"
    temp_output = "temp_output.jpg"
    
    try:
        image.save(temp_input)
        
        # Process image
        with st.spinner(f"Processing with {model_name.upper()}..."):
            result = pipeline.dehaze_single_image(temp_input, model_name, temp_output)
            
        if result["success"]:
            # Load processed image
            processed_image = Image.open(temp_output)
            
            # Clean up temporary files
            os.remove(temp_input)
            os.remove(temp_output)
            
            return processed_image, result
        else:
            st.error(f"Processing failed: {result['error']}")
            return None, result
            
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

def display_metrics(result):
    """Display processing metrics"""
    if result is None or not result.get("success"):
        return
        
    st.markdown("### 📊 Processing Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Processing Time",
            f"{result['processing_time']:.2f}s",
            delta=None
        )
    
    with col2:
        if result["metrics"].get("psnr"):
            st.metric(
                "PSNR",
                f"{result['metrics']['psnr']:.2f} dB",
                delta=None
            )
        else:
            st.metric("PSNR", "N/A", delta=None)
    
    with col3:
        if result["metrics"].get("ssim"):
            st.metric(
                "SSIM",
                f"{result['metrics']['ssim']:.4f}",
                delta=None
            )
        else:
            st.metric("SSIM", "N/A", delta=None)

def create_download_link(image, filename="dehazed_image.jpg"):
    """Create a download link for the processed image"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">📥 Download Image</a>'
    return href

def main():
    # Initialize session state
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = 'aodnet'
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    
    # Header
    st.markdown('<h1 class="main-header">🌫️ Image Dehazing Dashboard</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    Welcome to the Image Dehazing Dashboard! This tool helps remove fog, haze, and smog from images 
    using advanced deep learning models. Perfect for improving visibility in Indian winter conditions.
    """)
    
    # Load pipeline
    if st.session_state.pipeline is None:
        with st.spinner("Loading models... This may take a moment..."):
            st.session_state.pipeline = load_pipeline()
    
    if st.session_state.pipeline is None:
        st.error("Failed to initialize the dehazing pipeline. Please check the console for errors.")
        return
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Settings")
    
    # Model selection
    available_models = st.session_state.pipeline.get_model_info()["available_models"]
    selected_model = st.sidebar.selectbox(
        "Select Model",
        available_models,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
    )
    st.session_state.selected_model = selected_model
    
    # Display model information
    display_model_info(st.session_state.pipeline)
    
    # Main content area
    st.markdown("## 📸 Upload and Process Image")
    
    # File upload
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Input Image")
        uploaded_file = st.file_uploader(
            "Choose a hazy image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image with haze, fog, or smog"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("🚀 Dehaze Image", type="primary", use_container_width=True):
                processed_image, result = process_image(
                    st.session_state.pipeline, 
                    image, 
                    st.session_state.selected_model
                )
                
                if processed_image is not None:
                    st.session_state.processed_image = processed_image
                    st.session_state.processing_result = result
    
    with col2:
        st.markdown("### Dehazed Image")
        
        if 'processed_image' in st.session_state and st.session_state.processed_image is not None:
            st.image(
                st.session_state.processed_image, 
                caption=f"Dehazed with {st.session_state.selected_model.upper()}", 
                use_column_width=True
            )
            
            # Download link
            download_link = create_download_link(
                st.session_state.processed_image,
                f"dehazed_{st.session_state.selected_model}_{int(time.time())}.jpg"
            )
            st.markdown(download_link, unsafe_allow_html=True)
            
            # Display metrics
            if 'processing_result' in st.session_state:
                display_metrics(st.session_state.processing_result)
        else:
            st.info("Upload an image and click 'Dehaze Image' to see results")
    
    # Advanced features
    st.markdown("---")
    st.markdown("## 🔧 Advanced Features")
    
    # Batch processing
    with st.expander("📁 Batch Processing"):
        st.markdown("""
        Upload multiple images for batch processing. Results will be available for download as a ZIP file.
        """)
        
        batch_files = st.file_uploader(
            "Choose multiple images...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            key="batch_upload"
        )
        
        if batch_files and st.button("Process Batch", key="batch_process"):
            if len(batch_files) > 10:
                st.warning("Processing more than 10 images may take a while...")
            
            progress_bar = st.progress(0)
            batch_results = []
            
            for i, file in enumerate(batch_files):
                image = Image.open(file)
                processed_image, result = process_image(
                    st.session_state.pipeline,
                    image,
                    st.session_state.selected_model
                )
                
                if processed_image is not None:
                    batch_results.append({
                        'filename': file.name,
                        'image': processed_image,
                        'result': result
                    })
                
                progress_bar.progress((i + 1) / len(batch_files))
            
            if batch_results:
                st.success(f"Successfully processed {len(batch_results)} images!")
                # Here you could add ZIP download functionality
            else:
                st.error("No images were successfully processed.")
    
    # Model comparison
    with st.expander("⚖️ Model Comparison"):
        st.markdown("""
        Compare the performance of different models on the same image.
        """)
        
        if uploaded_file is not None:
            comparison_models = st.multiselect(
                "Select models to compare",
                available_models,
                default=[st.session_state.selected_model]
            )
            
            if len(comparison_models) > 1 and st.button("Compare Models", key="compare"):
                image = Image.open(uploaded_file)
                comparison_results = {}
                
                with st.spinner("Comparing models..."):
                    for model in comparison_models:
                        processed_image, result = process_image(
                            st.session_state.pipeline,
                            image,
                            model
                        )
                        if processed_image is not None:
                            comparison_results[model] = {
                                'image': processed_image,
                                'result': result
                            }
                
                # Display comparison results
                cols = st.columns(len(comparison_results))
                for i, (model, data) in enumerate(comparison_results.items()):
                    with cols[i]:
                        st.markdown(f"**{model.upper()}**")
                        st.image(data['image'], use_column_width=True)
                        if data['result']['metrics'].get('psnr'):
                            st.write(f"PSNR: {data['result']['metrics']['psnr']:.2f} dB")
                        if data['result']['metrics'].get('ssim'):
                            st.write(f"SSIM: {data['result']['metrics']['ssim']:.4f}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built for Indian winter conditions • Optimized for fog and smog removal</p>
        <p>Models: AOD-Net, DehazeNet, MSBDN • Metrics: PSNR, SSIM</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
