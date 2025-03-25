import streamlit as st
from PIL import Image

# Open the image
img = Image.open("Flux_Dev_A_serene_and_simplistic_scene_depicting_a_drone_hover_0.jpeg")

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

# Streamlit application
def main():
    
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

    st.markdown("""<style>
        .title2 {font-size:30px; color: #C52831; text-align: center; font-weight: bold;}
        .title1 {font-size:55px; color: #4A90E2; text-align: center; font-weight: bold;}
        .subheader {font-size: 25px; color: #2C3E50;}
    </style>""", unsafe_allow_html=True)

    st.markdown('<p class="title2">School of Aeronautical Engineering</p>', unsafe_allow_html=True)
    st.markdown('<p class="title1">Change Detection App</p>', unsafe_allow_html=True)
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

if __name__ == "__main__":
    main()
