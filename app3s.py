import streamlit as st
from PIL import Image

# Open the image
img = Image.open("photo_2025-03-25_15-47-07.jpg")

# Set the Streamlit page config
st.set_page_config(
    page_title="Change_detection",
    page_icon=img  # PIL image works fine here
)
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import base64
from skimage.metrics import structural_similarity as ssim


# Track objects across images with unique IDs
def track_objects_across_images(predictions1, predictions2, iou_threshold=0.2, score_threshold=0.3):
    """
    Track objects between two images and identify their status (added, removed, modified, unchanged)
    
    Args:
        predictions1: Object detection results from the first image
        predictions2: Object detection results from the second image
        iou_threshold: Threshold for considering objects as matching (lower = more lenient matching)
        score_threshold: Minimum confidence score to consider an object
        
    Returns:
        Dictionary of tracked objects with their status
    """
    tracked_objects = {}
    object_id = 0
    
    # First, filter objects based on confidence score
    boxes1 = []
    labels1 = []
    scores1 = []
    
    for i, box in enumerate(predictions1[0]['boxes']):
        score = predictions1[0]['scores'][i].cpu().item()
        if score > score_threshold:
            boxes1.append(box.int().cpu().numpy())
            labels1.append(predictions1[0]['labels'][i].cpu().item())
            scores1.append(score)
    
    boxes2 = []
    labels2 = []
    scores2 = []
    
    for i, box in enumerate(predictions2[0]['boxes']):
        score = predictions2[0]['scores'][i].cpu().item()
        if score > score_threshold:
            boxes2.append(box.int().cpu().numpy())
            labels2.append(predictions2[0]['labels'][i].cpu().item())
            scores2.append(score)
            
    print(f"After filtering: First image has {len(boxes1)} objects, Second image has {len(boxes2)} objects")
    
    # If either image has no objects after filtering, create dummy objects to ensure tracking works
    if len(boxes1) == 0 and len(boxes2) > 0:
        # Create a "dummy" object in the first image to make all second image objects "added"
        tracked_objects = {}
        for i, box in enumerate(boxes2):
            tracked_objects[i] = {
                'bbox': box,
                'status': 'added',
                'label': labels2[i],
                'score': scores2[i]
            }
        return tracked_objects
        
    if len(boxes2) == 0 and len(boxes1) > 0:
        # Create "dummy" objects in the second image to make all first image objects "removed"
        tracked_objects = {}
        for i, box in enumerate(boxes1):
            tracked_objects[i] = {
                'bbox': box,
                'status': 'removed',
                'label': labels1[i],
                'score': scores1[i]
            }
        return tracked_objects
    
    # Assign IDs to all objects in the first image
    for i, box in enumerate(boxes1):
        tracked_objects[object_id] = {
            'bbox': box,
            'status': 'existing',  # Initial status
            'label': labels1[i],
            'score': scores1[i]
        }
        object_id += 1
    
    # Match objects in the second image
    matched_indices = set()  # Track which objects in the second image have been matched
    
    # For each object in first image, find its match in second image
    for obj_id, obj_data in tracked_objects.items():
        best_match_idx = -1
        best_iou = iou_threshold  # Must exceed this threshold
        
        for j, box2 in enumerate(boxes2):
            if j in matched_indices:
                continue  # Skip already matched objects
                
            iou = calculate_iou(obj_data['bbox'], box2)
            if iou > best_iou:
                best_iou = iou
                best_match_idx = j
        
        if best_match_idx >= 0:
            # Found a match
            matched_indices.add(best_match_idx)
            box2 = boxes2[best_match_idx]
            label2 = labels2[best_match_idx]
            score2 = scores2[best_match_idx]
            
            # Determine if position or class changed
            if label2 != obj_data['label']:
                tracked_objects[obj_id]['status'] = 'modified_class'
                tracked_objects[obj_id]['new_label'] = label2
            elif np.sum(np.abs(obj_data['bbox'] - box2)) > 15:  # More lenient position change threshold
                tracked_objects[obj_id]['status'] = 'modified_position'
            else:
                tracked_objects[obj_id]['status'] = 'unchanged'
                
            # Update the bbox
            tracked_objects[obj_id]['bbox'] = box2
            tracked_objects[obj_id]['score'] = score2
        else:
            # No match found, object was removed
            tracked_objects[obj_id]['status'] = 'removed'
    
    # Add new objects (those in second image that weren't matched)
    for j, box2 in enumerate(boxes2):
        if j not in matched_indices:
            tracked_objects[object_id] = {
                'bbox': box2,
                'status': 'added',
                'label': labels2[j],
                'score': scores2[j]
            }
            object_id += 1
    
    return tracked_objects

# Visualize changes with color coding
def visualize_changes(image1_cv, image2_cv, tracked_objects):
    # Create copies of the images for visualization
    image1_vis = image1_cv.copy()
    image2_vis = image2_cv.copy()
    
    # Color mapping for different change statuses
    color_map = {
        'added': (0, 255, 0),      # Green
        'removed': (0, 0, 255),    # Red
        'modified_class': (255, 165, 0),  # Orange
        'modified_position': (255, 255, 0),  # Yellow
        'unchanged': (255, 255, 255),  # White
        'existing': (200, 200, 200)  # Light gray
    }
    
    # Visualize each object based on its status
    for obj_id, obj_data in tracked_objects.items():
        bbox = obj_data['bbox']
        status = obj_data['status']
        color = color_map[status]
        
        x1, y1, x2, y2 = bbox
        
        # Draw on appropriate image based on status
        if status in ['removed', 'existing']:
            cv2.rectangle(image1_vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image1_vis, f"ID:{obj_id} - {status}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if status in ['added', 'modified_class', 'modified_position', 'unchanged']:
            cv2.rectangle(image2_vis, (x1, y1), (x2, y2), color, 2)
            label_text = f"ID:{obj_id} - {status}"
            cv2.putText(image2_vis, label_text, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Create a composite visualization showing change distribution
    height, width = image1_cv.shape[:2]
    change_summary = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Count of each change type
    change_counts = {status: 0 for status in color_map.keys()}
    
    for obj_data in tracked_objects.values():
        change_counts[obj_data['status']] += 1
        
        # Draw the object on the change summary
        bbox = obj_data['bbox']
        status = obj_data['status']
        color = color_map[status]
        
        x1, y1, x2, y2 = bbox
        cv2.rectangle(change_summary, (x1, y1), (x2, y2), color, -1)  # Filled rectangle
    
    return image1_vis, image2_vis, change_summary, change_counts

# Load the pre-trained Faster R-CNN model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to(device)
    return model, device

# Preprocess images for alignment and resizing
def preprocess_images(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.resize(gray1, (640, 480))
    gray2 = cv2.resize(gray2, (640, 480))
    return gray1, gray2

# Resize images dynamically for performance
def resize_image(image, max_dim=800):
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        return cv2.resize(image, (int(w * scale), int(h * scale)))
    return image

# Improved change detection using SSIM
def change_detection_ssim(img1, img2, threshold=30):
    score, diff = ssim(img1, img2, full=True)
    diff = (diff * 255).astype("uint8")
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh, diff

# Perform object detection on the image
def detect_objects(image, model, device):
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    image_tensor = transform(image).to(device)
    with torch.no_grad():
        predictions = model([image_tensor])
    return predictions

# Calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Function to load and convert the image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Define page functions
def welcome_page():
    # CSS for welcome page
    st.markdown("""<style>
        .welcome-title {font-size:60px; color: #4A90E2; text-align: center; font-weight: bold; margin-bottom: 30px;}
        .welcome-subtitle {font-size:24px; color: #2C3E50; text-align: center; margin-bottom: 40px;}
        .feature-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .feature-title {font-size: 22px; font-weight: bold; color: #4A90E2; margin-bottom: 10px;}
        .get-started-btn {
            background-color: #4A90E2;
            color: white;
            font-size: 18px;
            padding: 12px 24px;
            border-radius: 5px;
            text-align: center;
            margin: 30px auto;
            display: block;
            width: 200px;
        }
    </style>""", unsafe_allow_html=True)
    
    # Display the school logo if available
    try:
        logo_path = r"logo (1).png" 
        logo_base64 = get_base64_of_bin_file(logo_path)
        st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/jpeg;base64,{logo_base64}" alt="logo" style="width:450px;">
        </div>
        """,
        unsafe_allow_html=True
        )
    except Exception:
        st.write("School of Aeronautical Engineering")
    
    # Welcome title and subtitle
    st.markdown('<p class="welcome-title">Welcome to Change Detection App</p>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-subtitle">Easily identify and analyze differences between images using advanced AI techniques</p>', unsafe_allow_html=True)
    
    # Features section with cards
    st.subheader("Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('''
        <div class="feature-card">
            <p class="feature-title">🔍 Object Detection</p>
            <p>Automatically detect objects in your images using state-of-the-art AI models</p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="feature-card">
            <p class="feature-title">📊 Change Analysis</p>
            <p>Get detailed statistics and visualizations about changes between images</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
        <div class="feature-card">
            <p class="feature-title">🔄 Object Tracking</p>
            <p>Track objects across images to identify what's been added, removed, or modified</p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="feature-card">
            <p class="feature-title">📈 Enhanced Visualization</p>
            <p>See changes highlighted with intuitive color coding and detailed annotations</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # How it works section
    st.subheader("How It Works")
    st.write("""
    1. **Upload Images** - Select two images you want to compare
    2. **Process Images** - Our AI analyzes both images and identifies all objects
    3. **Track Changes** - The application tracks objects across images and categorizes changes
    4. **Review Results** - View detailed reports and visualizations of all detected changes
    """)
    
    # Sample use cases
    st.subheader("Example Use Cases")
    st.write("""
    - **Surveillance & Security**: Monitor changes in security camera footage
    - **Environmental Monitoring**: Track changes in aerial or satellite imagery
    - **Quality Control**: Detect manufacturing defects or inconsistencies
    - **Research & Analysis**: Compare experimental results over time
    """)
    
    # Get started button
    if st.button("Get Started →", key="start_button"):
        st.session_state.page = "main_app"
        st.rerun()
    st.header("Preview of the Application", divider=True)

    st.subheader("Change Detection: Analyze changes in the image", divider=True)
    VIDEO_URL = "https://youtu.be/5MdvK1pK41o"
    st.video(VIDEO_URL, loop=True)


def main_app_page():
    # Original app code starts here
    st.markdown("""<style>
        .title1 {font-size:30px; color: #C52831; text-align: center; font-weight: bold;}
        .title {font-size:55px; color: #4A90E2; text-align: center; font-weight: bold;}
        .subheader {font-size: 25px; color: #2C3E50;}
    </style>""", unsafe_allow_html=True)

    # Add a "Back to Welcome" button at the top
    if st.button("← Back to Welcome Page", key="back_button"):
        st.session_state.page = "welcome"
        st.rerun()

    st.markdown('<p class="title1">School of Aeronautical Engineering</p>', unsafe_allow_html=True)
    st.markdown('<p class="title">Change Detection App</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Upload your images below for detection</p>', unsafe_allow_html=True)

    with st.expander("How to use the app"):
         st.write("""
            1. Upload two images you want to compare.
            2. After uploading both images, click the 'Detect Changes' button.
            3. Adjust the SSIM threshold to control the sensitivity of the change detection.
            4. Detected changes will be highlighted with text "new object".
        """)

    model, device = load_model()

    # Upload images
    uploaded_file1 = st.file_uploader("Upload First Image", type=["jpg", "jpeg", "png"])
    uploaded_file2 = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"])

    if uploaded_file1 is not None and uploaded_file2 is not None:
        # Read the uploaded images
        image1 = Image.open(uploaded_file1).convert("RGB")
        image2 = Image.open(uploaded_file2).convert("RGB")
        image1_cv = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
        image2_cv = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)

        # Resize images for processing
        image1_cv = resize_image(image1_cv)
        image2_cv = resize_image(image2_cv)

        # Preprocess the images
        gray1, gray2 = preprocess_images(image1_cv, image2_cv)

        # SSIM threshold slider
        threshold = st.slider("SSIM Threshold", min_value=0, max_value=255, value=30, step=5)

        # Button to detect changes
        if st.button("Detect Changes"):
            # Display a loading bar
            progress = st.progress(0)

            # Simulate progress as the processing takes place
            for i in range(100):
                progress.progress(i + 1)

            # Detect changes using SSIM
            change_mask, diff_map = change_detection_ssim(gray1, gray2, threshold)

            # Detect objects in both images
            predictions1 = detect_objects(image1_cv, model, device)
            predictions2 = detect_objects(image2_cv, model, device)

            # Draw bounding boxes on detected objects and identify new objects
            new_objects = []
            all_objects = []
            for bbox1 in predictions1[0]['boxes']:
                x1, y1, x2, y2 = bbox1.int().cpu().numpy()
                cv2.rectangle(image1_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green boxes for image 1

            for bbox2 in predictions2[0]['boxes']:
                x1, y1, x2, y2 = bbox2.int().cpu().numpy()
                new = True
                for bbox1 in predictions1[0]['boxes']:
                    if calculate_iou(bbox1.int().cpu().numpy(), bbox2.int().cpu().numpy()) > 0.4:
                        new = False
                        break
                if new:
                    new_objects.append([x1, y1, x2, y2])
                    cv2.rectangle(image2_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Red boxes for new objects
                all_objects.append([x1, y1, x2, y2])

            # Display results
            st.subheader("Change Mask (Thresholded)")
            st.image(change_mask, channels="GRAY")

            st.subheader("SSIM Difference Map")
            st.image(diff_map, channels="GRAY")

            st.subheader("Detected Objects in First Image")
            st.image(cv2.cvtColor(image1_cv, cv2.COLOR_BGR2RGB))

            st.subheader("Detected Objects in Second Image")
            st.image(cv2.cvtColor(image2_cv, cv2.COLOR_BGR2RGB))

            # Composite image with new objects
            composite_image = image2_cv.copy()
            for bbox in new_objects:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(composite_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(composite_image, "New Object", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            st.subheader("Composite Image with New Objects")
            st.image(cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB))

            # List bounding boxes of new objects
            if new_objects:
                st.write(f"Detected {len(new_objects)} new objects:")
                for idx, bbox in enumerate(new_objects):
                    st.write(f"Object {idx + 1}: Bounding Box = {bbox}")
            else:
                st.write("No new objects detected.")

        # Add a new button for advanced change detection
        if st.button("Detect Changes with Tracking"):
            # Display a loading bar
            progress = st.progress(0)
            
            # Simulate progress as the processing takes place
            for i in range(30):
                progress.progress(i + 1)
            
            # Make sure images are loaded and properly processed
            if 'image1_cv' not in locals() or 'image2_cv' not in locals():
                st.error("Please ensure both images are uploaded properly.")
                return
                
            # Ensure the preprocessing is done
            gray1, gray2 = preprocess_images(image1_cv, image2_cv)
            
            # Detect changes using SSIM first
            change_mask, diff_map = change_detection_ssim(gray1, gray2, threshold)
            
            progress.progress(40)
            
            # Detect objects in both images
            with st.spinner("Detecting objects..."):
                predictions1 = detect_objects(image1_cv, model, device)
                predictions2 = detect_objects(image2_cv, model, device)
            
            # Show debug information about detections
            st.write(f"Objects detected in first image: {len(predictions1[0]['boxes'])}")
            st.write(f"Objects detected in second image: {len(predictions2[0]['boxes'])}")
            
            # Show confidence scores for better debugging
            st.write("First image object confidence scores:")
            scores1 = [f"{score:.2f}" for score in predictions1[0]['scores'].cpu().numpy()]
            st.write(", ".join(scores1[:10]) + ("..." if len(scores1) > 10 else ""))
            
            st.write("Second image object confidence scores:")
            scores2 = [f"{score:.2f}" for score in predictions2[0]['scores'].cpu().numpy()]
            st.write(", ".join(scores2[:10]) + ("..." if len(scores2) > 10 else ""))
            
            progress.progress(60)
            
            # Use a very low IoU threshold and confidence threshold to ensure matches
            tracked_objects = track_objects_across_images(
                predictions1, predictions2, 
                iou_threshold=0.2,  # Lower threshold for matching
                score_threshold=0.3  # Lower threshold for considering objects
            )
            
            progress.progress(75)
            
            # Check if we got any tracked objects
            if not tracked_objects:
                st.warning("No objects were successfully tracked between images. The images may be too different or the detection model may not be identifying the same objects.")
                # Show the raw images side by side for comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(image1_cv, cv2.COLOR_BGR2RGB), caption="First Image")
                with col2:
                    st.image(cv2.cvtColor(image2_cv, cv2.COLOR_BGR2RGB), caption="Second Image")
                return
                
            # Visualize changes
            img1_vis, img2_vis, change_summary, change_counts = visualize_changes(
                image1_cv.copy(), image2_cv.copy(), tracked_objects)
            
            # Final progress update
            progress.progress(100)
            
            # Display results
            st.subheader("Object Tracking Results")
            
            # Display tracked images
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(img1_vis, cv2.COLOR_BGR2RGB), caption="First Image with Tracked Objects")
            with col2:
                st.image(cv2.cvtColor(img2_vis, cv2.COLOR_BGR2RGB), caption="Second Image with Tracked Objects")
            
            # Display change summary
            st.subheader("Change Classification Summary")
            
            # Create columns for the summary statistics
            stat_cols = st.columns(5)
            with stat_cols[0]:
                st.metric("Added", change_counts['added'], None, delta_color="normal")
            with stat_cols[1]:
                st.metric("Removed", change_counts['removed'], None, delta_color="normal")
            with stat_cols[2]:
                st.metric("Modified Class", change_counts['modified_class'], None, delta_color="normal")
            with stat_cols[3]:
                st.metric("Modified Position", change_counts['modified_position'], None, delta_color="normal")
            with stat_cols[4]:
                st.metric("Unchanged", change_counts['unchanged'], None, delta_color="normal")
            
            # Display change distribution visualization
            st.image(cv2.cvtColor(change_summary, cv2.COLOR_BGR2RGB), caption="Change Distribution")
            
            # Display detailed change report
            st.subheader("Detailed Change Report")
            
            # Create an expandable section for the detailed report
            with st.expander("View Object-by-Object Report"):
                # Group objects by change type
                change_groups = {'added': [], 'removed': [], 'modified_class': [], 
                                'modified_position': [], 'unchanged': []}
                
                for obj_id, obj_data in tracked_objects.items():
                    change_groups[obj_data['status']].append((obj_id, obj_data))
                
                # Display groups
                if change_groups['added']:
                    st.write("### Added Objects")
                    for obj_id, obj_data in change_groups['added']:
                        st.write(f"- Object ID {obj_id}: Added at position {obj_data['bbox']}")
                
                if change_groups['removed']:
                    st.write("### Removed Objects")
                    for obj_id, obj_data in change_groups['removed']:
                        st.write(f"- Object ID {obj_id}: Removed from position {obj_data['bbox']}")
                
                if change_groups['modified_class']:
                    st.write("### Objects with Modified Class")
                    for obj_id, obj_data in change_groups['modified_class']:
                        if 'new_label' in obj_data:
                            st.write(f"- Object ID {obj_id}: Class changed from {obj_data['label']} to {obj_data['new_label']}")
                        else:
                            st.write(f"- Object ID {obj_id}: Class modified (original class: {obj_data['label']})")
                
                if change_groups['modified_position']:
                    st.write("### Objects with Modified Position")
                    for obj_id, obj_data in change_groups['modified_position']:
                        st.write(f"- Object ID {obj_id}: Position/size changed at {obj_data['bbox']}")

# Main function to run the app
def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "welcome"
    
    # Display current page
    if st.session_state.page == "welcome":
        welcome_page()
    else:
        main_app_page()

if __name__ == "__main__":
    main()
