"""
Satellite Image Enhancement System - Premium UI
"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import io
import time

from inference_advanced import AdvancedInferenceEngine

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Satellite Images",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. PREMIUM CSS STYLING
st.markdown("""
<style>
    /* Main Background & Font */
    .stApp {
        background: linear-gradient(to bottom right, #f8f9fa, #e9ecef);
        color: #2c3e50;
    }
    
    /* Headers */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        letter-spacing: -1px;
        background: -webkit-linear-gradient(45deg, #1a237e, #0d47a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        color: #546e7a;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }

    /* Cards/Containers */
    .css-1r6slb0, .stMarkdown {
        border-radius: 15px;
    }
    
    /* Image Containers */
    .image-container {
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        overflow: hidden;
        transition: transform 0.3s ease;
    }
    .image-container:hover {
        transform: translateY(-5px);
    }
    
    /* Custom Button */
    .stButton>button {
        background: linear-gradient(90deg, #2196F3, #1976D2);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 30px;
        font-weight: 600;
        letter-spacing: 0.5px;
        width: 100%;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1976D2, #1565C0);
        box-shadow: 0 6px 20px rgba(33, 150, 243, 0.4);
        transform: scale(1.02);
    }

    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #2196F3 !important;
    }
    
    /* Footer */
    footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    return AdvancedInferenceEngine(device="cpu")

def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def main():
    # HEADER
    st.markdown("<h3>Atmospheric Noise-Aware Preprocessing for Accurate Change Detection in Satellite Imagery</h3>", unsafe_allow_html=True)
    
    engine = load_engine()
    
    # UPLOAD SECTION
    with st.container():
        col_up_1, col_up_2, col_up_3 = st.columns([1, 2, 1])
        with col_up_2:
            uploaded_files = st.file_uploader(
                "Drop your image here",
                type=['png', 'jpg', 'jpeg', 'tif'],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )

    if uploaded_files:
        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

        # --- IMMEDIATE PREVIEW: show uploaded image(s) right after upload ---
        st.markdown("<h4 style='text-align: center; color: #34495e;'>Preview of Uploaded Image(s)</h4>", unsafe_allow_html=True)
        # If multiple files, display them in a responsive grid
        max_preview_cols = min(len(uploaded_files), 4)
        preview_cols = st.columns(max_preview_cols)

        for idx, f in enumerate(uploaded_files):
            try:
                pil_img = Image.open(f)
            except Exception:
                # skip files that cannot be opened as images
                continue

            col = preview_cols[idx % max_preview_cols]
            with col:
                st.image(pil_img, caption=f.name, width=300, output_format="PNG")

        
        # PROCESS BUTTON
        col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 1, 1])
        with col_btn_2:
            process_btn = st.button("✨ ENHANCE IMAGE", use_container_width=True)

        if process_btn:
            num_images = len(uploaded_files)
            
            # ---------------------------------------------------------
            # SINGLE IMAGE MODE
            # ---------------------------------------------------------
            if num_images == 1:
                image_file = uploaded_files[0]
                pil_image = Image.open(image_file)
                cv2_image = pil_to_cv2(pil_image)
                
                with st.spinner("Running Deep Learning Pipeline..."):
                    start_time = time.time()
                    results = engine.process_single_image(cv2_image)
                    elapsed = time.time() - start_time
                
                # --- RESULTS DISPLAY ---
                st.markdown("---")
                
                # 3-Column Layout: Original | Disturbances Marked | Enhanced
                col1, col2, col3 = st.columns(3, gap="medium")
                
                with col1:
                    st.markdown("<h4 style='text-align: center; color: #7f8c8d;'>ORIGINAL</h4>", unsafe_allow_html=True)
                    st.image(pil_image, use_container_width=True, output_format="PNG")
                    st.markdown(f"<div style='text-align: center; font-size: 0.8rem; color: #95a5a6'>Resolution: {pil_image.size[0]}x{pil_image.size[1]}</div>", unsafe_allow_html=True)

                with col2:
                    st.markdown("<h4 style='text-align: center; color: #e74c3c;'>DISTURBANCES DETECTED</h4>", unsafe_allow_html=True)
                    if 'disturbance_visualization' in results:
                        disturbance_pil = cv2_to_pil(results['disturbance_visualization'])
                        st.image(disturbance_pil, use_container_width=True, output_format="PNG")
                    else:
                        st.image(pil_image, use_container_width=True, output_format="PNG")
                    st.markdown("<div style='text-align: center; font-size: 0.8rem; color: #e74c3c'>Red=Shadow, Yellow=Haze</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("<h4 style='text-align: center; color: #27ae60;'>ENHANCED</h4>", unsafe_allow_html=True)
                    enhanced_pil = cv2_to_pil(results['processed'])
                    st.image(enhanced_pil, use_container_width=True, output_format="PNG")
                    st.markdown(f"<div style='text-align: center; font-size: 0.8rem; color: #27ae60'>Processed in {elapsed:.2f}s</div>", unsafe_allow_html=True)

                # COMPREHENSIVE AI REPORT
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center; color: #2c3e50;'>📊 Complete AI Analysis Report</h3>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                # 1. AI CAPTION
                if 'caption' in results:
                    st.markdown("### 🖼️ Image Caption")
                    st.markdown(f"<div style='background: #e3f2fd; padding: 15px; border-radius: 10px; margin: 10px 0; font-size: 1.1em;'><strong>{results['caption']}</strong></div>", unsafe_allow_html=True)
                
                # 2. QUALITY ISSUES DETECTED
                if 'detection_raw' in results:
                    st.markdown("### ⚠️ Quality Issues Detected")
                    detection = results['detection_raw']
                    issues = []
                    
                    # Shadow
                    if 'shadow' in detection and detection['shadow'].get('shadow_pct', 0) > 5:
                        shadow_pct = detection['shadow']['shadow_pct']
                        issues.append(f"🌑 **Shadow:** {shadow_pct:.1f}% of image")
                    
                    # Haze
                    if 'haze' in detection and detection['haze'].get('haze_score', 0) > 20:
                        haze_score = detection['haze']['haze_score']
                        issues.append(f"🌫️ **Haze/Low Contrast:** Score {haze_score:.1f}")
                    
                    # Cloud
                    if 'cloud' in detection and detection['cloud'].get('cloud_pct', 0) > 1:
                        cloud_pct = detection['cloud']['cloud_pct']
                        issues.append(f"☁️ **Clouds:** {cloud_pct:.1f}% coverage")
                    
                    if issues:
                        st.markdown("<div style='background: #fff3e0; padding: 15px; border-radius: 10px; margin: 10px 0;'>" + "<br>".join(issues) + "</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='background: #e8f5e9; padding: 15px; border-radius: 10px; margin: 10px 0;'>✅ <strong>No major quality issues detected</strong></div>", unsafe_allow_html=True)
                
                # 4. ENHANCEMENTS APPLIED
                if 'applied_enhancements' in results and results['applied_enhancements']:
                    st.markdown("### ✨ Enhancements Applied")
                    enhancements = results['applied_enhancements']
                    enh_list = []
                    for enh in enhancements:
                        enh_name = enh.replace('_', ' ').title()
                        enh_list.append(f"✓ {enh_name}")
                    
                    st.markdown("<div style='background: #e8f5e9; padding: 15px; border-radius: 10px; margin: 10px 0;'>" + "<br>".join(enh_list) + "</div>", unsafe_allow_html=True)
                
                # 5. DETAILED EXPLANATIONS
                # if 'explanations' in results and results['explanations']:
                #     st.markdown("### 📝 Detailed Processing Log")
                #     for explanation in results['explanations']:
                #         st.markdown(f"<div style='background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 5px 0; font-size: 0.9em;'>{explanation}</div>", unsafe_allow_html=True)
                
                # DOWNLOAD
                st.markdown("<br>", unsafe_allow_html=True)
                col_d_1, col_d_2, col_d_3 = st.columns([1, 1, 1])
                with col_d_2:
                    buf = io.BytesIO()
                    enhanced_pil.save(buf, format='PNG')
                    st.download_button(
                        label="⬇️ Download Result",
                        data=buf.getvalue(),
                        file_name=f"enhanced_{image_file.name}",
                        mime="image/png",
                        use_container_width=True
                    )

            # ---------------------------------------------------------
            # DUAL IMAGE MODE
            # ---------------------------------------------------------
            # ---------------------------------------------------------
            # DUAL IMAGE MODE (COMPARISON)
            # ---------------------------------------------------------
            elif num_images == 2:
                img1 = pil_to_cv2(Image.open(uploaded_files[0]))
                img2 = pil_to_cv2(Image.open(uploaded_files[1]))
                
                # Process both images and compare
                with st.spinner("Processing and comparing images..."):
                    comparison = engine.process_dual_images(img1, img2)
                
                st.markdown("---")
                
                # Extract results
                res1 = comparison['image1']
                res2 = comparison['image2']
                change_data = comparison['change_detection']
                
                # Layout: [ Image 1 ] [ CHANGE MAP ] [ Image 2 ]
                col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
                
                with col1:
                    st.markdown("<h4 style='text-align: center; color: #7f8c8d;'>Image 1 (Enhanced)</h4>", unsafe_allow_html=True)
                    st.image(cv2_to_pil(res1['processed']), use_container_width=True)
                
                with col2:
                    st.markdown("<h4 style='text-align: center; color: #e74c3c;'>Detected Changes</h4>", unsafe_allow_html=True)
                    
                    # Create a nice overlay instead of raw mask
                    mask = change_data['mask']
                    
                    # Make mask Magenta where changes are
                    overlay = res2['processed'].copy()
                    
                    # Ensure mask is scaled
                    if mask.shape[:2] != overlay.shape[:2]:
                        mask = cv2.resize(mask, (overlay.shape[1], overlay.shape[0]))
                        
                    # Draw Magenta contours
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours, -1, (255, 0, 255), 2)
                    
                    # Semi-transparent fill
                    overlay_layer = overlay.copy()
                    overlay_layer[mask > 0] = [255, 0, 255] # Magenta
                    cv2.addWeighted(overlay_layer, 0.3, overlay, 0.7, 0, overlay)
                    
                    change_pct = change_data['stats']['change_percentage']
                    st.image(cv2_to_pil(overlay), caption=f"Magenta = Changes ({change_pct:.2f}%)", use_container_width=True)
                
                with col3:
                    st.markdown("<h4 style='text-align: center; color: #27ae60;'>Image 2 (Enhanced)</h4>", unsafe_allow_html=True)
                    st.image(cv2_to_pil(res2['processed']), use_container_width=True)
                
                # COMPREHENSIVE DUAL-IMAGE REPORT
                st.markdown("<br>", unsafe_allow_html=True)
                # st.markdown("<h3 style='text-align: center; color: #2c3e50;'>📊 Complete Comparison Report</h3>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                # CHANGE DETECTION SUMMARY
                st.markdown("### 🔄 Change Detection Summary")
                change_summary = f"**{change_pct:.2f}%** of the image area shows structural changes between the two time periods."
                if change_pct > 30:
                    change_summary += " (Major changes detected)"
                elif change_pct > 10:
                    change_summary += " (Moderate changes detected)"
                else:
                    change_summary += " (Minor changes detected)"
                st.markdown(f"<div style='background: #e3f2fd; padding: 15px; border-radius: 10px; margin: 10px 0; font-size: 1.1em;'><strong>{change_summary}</strong></div>", unsafe_allow_html=True)
                
                # IMAGE 1 ANALYSIS
                st.markdown("### 📷 Image 1 Analysis")
                res1 = comparison['image1']
                
                # Detailed analysis in expandable sections
                with st.expander("🔍 Detected Objects & Scene", expanded=True):
                    if 'caption' in res1:
                        st.markdown(f"**AI Caption:** {res1['caption']}")
                
                with st.expander("⚠️ Quality Issues Detected", expanded=True):
                    if 'detection_raw' in res1:
                        det1 = res1['detection_raw']
                        issues_found = []
                        
                        if 'shadow' in det1:
                            shadow_pct1 = det1['shadow'].get('shadow_pct', 0)
                            if shadow_pct1 > 1:
                                issues_found.append(f"🌑 **Shadow:** {shadow_pct1:.1f}% coverage (detected in LAB color space)")
                        
                        if 'haze' in det1:
                            haze_score1 = det1['haze'].get('haze_score', 0)
                            if haze_score1 > 10:
                                issues_found.append(f"🌫️ **Haze:** Score {haze_score1:.1f} (low contrast detection)")
                        
                        if issues_found:
                            for issue in issues_found:
                                st.markdown(issue)
                        else:
                            st.success("✅ No major quality issues detected")
                
                with st.expander("✨ Enhancements Applied", expanded=True):
                    if 'applied_enhancements' in res1 and res1['applied_enhancements']:
                        for enh in res1['applied_enhancements']:
                            st.markdown(f"✓ **{enh}**")
                        
                        # Show algorithm details
                        if 'explanations' in res1 and res1['explanations']:
                            st.markdown("**Algorithms Used:**")
                            for exp in res1['explanations']:
                                st.markdown(f"- {exp}")
                    else:
                        st.info("No enhancements needed")
                
                # IMAGE 2 ANALYSIS
                st.markdown("### 📷 Image 2 Analysis")
                res2_data = comparison['image2']
                
                with st.expander("🔍 Detected Objects & Scene", expanded=True):
                    if 'caption' in res2_data:
                        st.markdown(f"**AI Caption:** {res2_data['caption']}")
                
                with st.expander("⚠️ Quality Issues Detected", expanded=True):
                    if 'detection_raw' in res2_data:
                        det2 = res2_data['detection_raw']
                        issues_found = []
                        
                        if 'shadow' in det2:
                            shadow_pct2 = det2['shadow'].get('shadow_pct', 0)
                            if shadow_pct2 > 1:
                                issues_found.append(f"🌑 **Shadow:** {shadow_pct2:.1f}% coverage (detected in LAB color space)")
                        
                        if 'haze' in det2:
                            haze_score2 = det2['haze'].get('haze_score', 0)
                            if haze_score2 > 10:
                                issues_found.append(f"🌫️ **Haze:** Score {haze_score2:.1f} (low contrast detection)")
                        
                        if issues_found:
                            for issue in issues_found:
                                st.markdown(issue)
                        else:
                            st.success("✅ No major quality issues detected")
                
                with st.expander("✨ Enhancements Applied", expanded=True):
                    if 'applied_enhancements' in res2_data and res2_data['applied_enhancements']:
                        for enh in res2_data['applied_enhancements']:
                            st.markdown(f"✓ **{enh}**")
                        
                        # Show algorithm details
                        if 'explanations' in res2_data and res2_data['explanations']:
                            st.markdown("**Algorithms Used:**")
                            for exp in res2_data['explanations']:
                                st.markdown(f"- {exp}")
                    else:
                        st.info("No enhancements needed")
                
                # SPECIFIC DIFFERENCES
                st.markdown("### 🔍 Specific Differences Between Images")
                
                # Calculate specific differences
                diff_details = []
                
                if 'detection_raw' in res1 and 'detection_raw' in res2_data:
                    det1 = res1['detection_raw']
                    det2 = res2_data['detection_raw']
                    
                    # Shadow difference
                    if 'shadow' in det1 and 'shadow' in det2:
                        shadow1 = det1['shadow'].get('shadow_pct', 0)
                        shadow2 = det2['shadow'].get('shadow_pct', 0)
                        shadow_diff = abs(shadow2 - shadow1)
                        
                        if shadow_diff > 2:
                            if shadow2 > shadow1:
                                diff_details.append(f"🌑 **Shadow increased:** +{shadow_diff:.1f}% in Image 2 (likely different sun angle)")
                            else:
                                diff_details.append(f"🌑 **Shadow decreased:** -{shadow_diff:.1f}% in Image 2")
                    
                    # Haze difference
                    if 'haze' in det1 and 'haze' in det2:
                        haze1 = det1['haze'].get('haze_score', 0)
                        haze2 = det2['haze'].get('haze_score', 0)
                        haze_diff = abs(haze2 - haze1)
                        
                        if haze_diff > 5:
                            if haze2 > haze1:
                                diff_details.append(f"🌫️ **Haze increased:** +{haze_diff:.1f} in Image 2 (atmospheric conditions changed)")
                            else:
                                diff_details.append(f"🌫️ **Haze decreased:** -{haze_diff:.1f} in Image 2")
                
                # Structural changes
                diff_details.append(f"🏗️ **Structural changes:** {change_pct:.2f}% of image area")
                
                if diff_details:
                    for detail in diff_details:
                        st.markdown(f"- {detail}")
                else:
                    st.info("No significant differences detected")
                
                # DETAILED SEMANTIC ANALYSIS
                if 'semantic_analysis' in comparison and comparison['semantic_analysis']:
                    st.markdown("### 📊 Detailed Semantic Analysis")
                    st.markdown(f"\u003cdiv style='background: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;'\u003e{comparison['semantic_analysis']}\u003c/div\u003e", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
