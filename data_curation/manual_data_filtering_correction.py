import streamlit as st
import json
import os
import glob
import math
from PIL import Image
import io
import hydra
from omegaconf import DictConfig
import logging
from pixie.utils import resolve_paths, load_json, save_json, set_logger

# Configuration constants
COLUMNS_PER_ROW = 12  # Keep the original number of columns

# Set page configuration for wider layout
st.set_page_config(layout="wide")

# Custom CSS to make checkboxes smaller and images larger
st.markdown("""
<style>
    /* Make checkboxes more compact */
    .stCheckbox {
        padding: 0 !important;
        margin: 0 !important;
        min-height: 20px !important;
    }
    /* Remove extra padding around checkbox */
    .stCheckbox > label {
        padding: 0 !important;
    }
    /* Make checkbox text smaller */
    .stCheckbox > label > p {
        font-size: 0.7rem !important;
    }
    /* Add some spacing between images */
    .image-grid {
        margin: 2px;
        border: 1px solid #eee;
        padding: 2px;
    }
    /* Force container to expand to full width */
    .main .block-container {
        max-width: 100% !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    /* Ensure images expand properly */
    .stImage > img {
        width: 100% !important;
        object-fit: contain !important;
    }
    /* Make columns have less padding */
    .row-widget > div {
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)


def find_image_path(obj_id, render_base_dir, save_folder):
    """
    Given an object ID like "flowers/1d419eb4c9d34fc1a7f1e88d6ea8d55a/",
    return a .png path if found in render directory.
    """
    # 1) Direct guess: "class_render_outputs/flowers/1d419e.../000.png"
    expected_path = os.path.join(render_base_dir, save_folder, obj_id, "000.png")
    if os.path.exists(expected_path):
        return expected_path
    
    # 2) Try the first PNG in that folder
    obj_dir = os.path.join(render_base_dir, save_folder, obj_id)
    if os.path.exists(obj_dir):
        png_files = glob.glob(f"{obj_dir}/*.png")
        if png_files:
            return png_files[0]
    
    # 3) Fallback: walk the entire directory
    for root, dirs, files in os.walk(os.path.join(render_base_dir, save_folder)):
        for file in files:
            if file.endswith(".png") and obj_id.replace("/", os.sep) in root:
                return os.path.join(root, file)

    return None

def display_image_grid(items, flip_label, flip_prefix, render_base_dir, save_folder):
    """
    Display a grid of images with checkboxes in a more efficient layout
    """
    # Calculate how many rows we need based on the number of items
    n_rows = math.ceil(len(items) / COLUMNS_PER_ROW)
    
    for row_idx in range(n_rows):
        # Create a row with the specified number of columns
        cols = st.columns(COLUMNS_PER_ROW, gap="small")
        
        # Get items for this row
        start_idx = row_idx * COLUMNS_PER_ROW
        end_idx = min(start_idx + COLUMNS_PER_ROW, len(items))
        row_items = items[start_idx:end_idx]
        
        # Display each item in its column
        for col_idx, (obj_id, data) in enumerate(row_items):
            if col_idx < len(cols):  # Make sure we have a column for this item
                with cols[col_idx]:
                    img_path = find_image_path(obj_id, render_base_dir, save_folder)
                    if img_path:
                        # Get a truncated ID for display
                        short_id = obj_id.split('/')[-2][:8] if '/' in obj_id else obj_id[:8]
                        
                        # Display the image and make it take up more space
                        st.markdown(f'<div class="image-grid">', unsafe_allow_html=True)
                        
                        # Create a unique key for this image's expander
                        expander_key = f"expander_{obj_id}"
                        
                        # Display the image
                        st.image(img_path, use_container_width=True)
                        
                        # Add an expander for details
                        with st.expander("Details", expanded=False):
                            # Show object ID in a copyable code block
                            st.markdown("**Full Object ID:**")
                            st.code(obj_id, language="text")
                            
                            st.markdown("**Image Path:**")
                            st.code(img_path, language="text")
                            
                            if "reason" in data:
                                st.markdown(f"**VLM Reason:** {data['reason']}")
                            if "error" in data:
                                st.markdown(f"**Error:** {data['error']}")
                        
                        # Display a compact checkbox below the image
                        flip_key = f"{flip_prefix}_{obj_id}"
                        col1, col2 = st.columns([1, 5])
                        with col1:
                            st.checkbox(
                                "",
                                key=flip_key,
                                value=False,
                                label_visibility="collapsed"
                            )
                        with col2:
                            st.caption(short_id)  # Show truncated ID instead
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error(f"Image not found for {obj_id}")

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # Set up logging
    set_logger()
    
    # Resolve paths
    cfg = resolve_paths(cfg)
    
    # Get configuration
    correction_cfg = cfg.data_curation.manual_correction
    assert correction_cfg.obj_class, "obj_class must be specified for manual correction"
    
    # Set up paths
    json_path = os.path.join(cfg.paths.vlm_filtering_results_dir, 
                            correction_cfg.obj_class, 
                            correction_cfg.input_file)
    new_json_path = os.path.join(cfg.paths.vlm_filtering_results_dir, 
                                correction_cfg.obj_class, 
                                correction_cfg.output_file)
    
    st.title("Image Filtering")

    if not os.path.exists(json_path):
        st.error(f"JSON file not found at {json_path}")
        return

    # Load the JSON:  { "obj_id": {"is_appropriate": bool, ...}, ... }
    filtered_results = load_json(json_path)

    # Separate into inappropriate vs. appropriate
    inappropriate_items = [
        (obj_id, data) for obj_id, data in filtered_results.items() 
        if data.get("is_appropriate") == False
    ]
    appropriate_items = [
        (obj_id, data) for obj_id, data in filtered_results.items() 
        if data.get("is_appropriate") == True
    ]

    # Stats at the top
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Images", len(filtered_results))
    with col2:
        st.metric("Discarded Images", len(inappropriate_items))
    with col3:
        st.metric("Chosen Images", len(appropriate_items))

    st.markdown("""
        <div style="background-color:#f0f2f6; padding:10px; border-radius:5px; margin-bottom:10px">
        <h3 style="margin-top:0">Instructions</h3>
        <ul>
          <li><b>Click "Details"</b> below any image to see its full details and object ID</li>
          <li><b>Click the code block</b> to copy the object ID</li>
          <li><b>Check the box</b> below any image you want to flip to the other category</li>
          <li>All changes will be applied when you click <b>"Save Changes"</b> at the bottom</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

    # We'll use a single form so there's only one "Save Changes" button
    with st.form("flip_form"):
        # Images are now displayed directly in the form instead of in tabs
        # This ensures all images are visible at once and maximizes horizontal space
        
        # Discarded Images Section
        st.subheader(f"Discarded Images ({len(inappropriate_items)})")
        st.markdown("<p style='color:gray; margin-top:-10px'>Check boxes to flip to 'Chosen'</p>", unsafe_allow_html=True)
        
        display_image_grid(
            inappropriate_items, 
            "", # Empty label as we're using collapsed visibility
            "flip_to_appropriate",
            cfg.paths.render_outputs_base_dir,
            correction_cfg.obj_class,
        )
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Chosen Images Section
        st.subheader(f"Chosen Images ({len(appropriate_items)})")
        st.markdown("<p style='color:gray; margin-top:-10px'>Check boxes to flip to 'Discarded'</p>", unsafe_allow_html=True)
        
        display_image_grid(
            appropriate_items, 
            "", # Empty label as we're using collapsed visibility
            "flip_to_inappropriate",
            cfg.paths.render_outputs_base_dir,
            correction_cfg.obj_class,
        )

        # ------------------
        # Save Changes Button
        # ------------------
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("Save Changes", use_container_width=True)
        
        if submitted:
            changes_made = 0
            
            # 1) For each item in the "inappropriate" list:
            for obj_id, data in inappropriate_items:
                flip_key = f"flip_to_appropriate_{obj_id}"
                if st.session_state.get(flip_key, False):
                    filtered_results[obj_id]["is_appropriate"] = True
                    changes_made += 1

            # 2) For each item in the "appropriate" list:
            for obj_id, data in appropriate_items:
                flip_key = f"flip_to_inappropriate_{obj_id}"
                if st.session_state.get(flip_key, False):
                    filtered_results[obj_id]["is_appropriate"] = False
                    changes_made += 1

            print("changes_made", changes_made)
            # Only save if changes were made
            if changes_made > 0:
                save_json(filtered_results, new_json_path)
                st.success(f"Changes saved to JSON! ({changes_made} images flipped)")
            else:
                st.info("No changes were made.")

if __name__ == "__main__":
    main()