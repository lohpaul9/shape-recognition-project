import cv2
import numpy as np
from sklearn.cluster import KMeans

debug_mode = False  # Set this to True to enable debug mode
AREA_THRESHOLD_DEFAULT = 18000 # Default area threshold for filtering contours

def show_image(image, window_name):
    """
    Displays an image in a window.
    
    Parameters:
    - image: The image to be displayed.
    - window_name: The name of the window.
    """
    if debug_mode:
        cv2.imshow(window_name, image)
        cv2.waitKey(0)

def process_frame(frame):
    """
    Processes a single frame to detect shapes against a textured background.
    We base this off key observations:
    1. The shapes are smooth, the background is usually textured like grass or uneven concrete
    2. The shapes are a different color from the background. 
    The background is a uniform color. The shapes are usually a different color but might have a gradual color gradient. 
    3. The shapes do not have an outline. They end where the background begins. 

    Parameters:
    - frame: The input frame (color image).

    Returns:
    - output_frame: The processed frame with contours and coordinates overlayed.
    """
    # Step 1: Convert to LAB color space for better color segmentation
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab_frame)

    show_image(l, 'lab')

    # Step 2: Apply a Gaussian blur to smooth gradients and reduce noise from the background
    blurred = cv2.GaussianBlur(l, (5, 5), 0)

    show_image(blurred, 'blurred')

    # Step 3: Adaptive thresholding to isolate shapes from background based on brightness
    # We use adaptive thresholding because each shape has a different local brightness relative to background
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, blockSize=11, C=2)

    show_image(binary, 'thresholding')

    # Step 4: Morphological operations to clean up the texture noise in the background
    # This really helps to remove the small 'shapes' formed by textures in the background
    # Note that setting kernel to too large will result in loss of shape edges
    kernel = np.ones((7, 7), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=3)

    show_image(opened, 'opened')

    # Step 5: Detect edges in the cleaned binary image
    edges = cv2.Canny(opened, 50, 150)

    show_image(edges, 'edges')

    # Step 6: Dilate the edges to close gaps and smooth contours
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    show_image(dilated, 'dilated')

    # Step 7: Find contours and filter them based on area threshold
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filter by area threshold
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > AREA_THRESHOLD_DEFAULT]
    if len(filtered_contours) == 0:
        filtered_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Step 8: Create an output frame to draw results on
    output_frame = frame.copy()
    
    for cnt in filtered_contours:
        # Calculate moments to find the center of the shape
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Draw the center, contour, and coordinates
            cv2.circle(output_frame, (cX, cY), 5, (0, 0, 255), -1)
            cv2.drawContours(output_frame, [cnt], -1, (0, 255, 0), 2)
            # Overlay the coordinates
            cv2.putText(output_frame, f"({cX}, {cY})", (cX - 40, cY - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return output_frame


def process_video(input_video_path):
    """
    Processes a video file, applying shape detection to each frame, displaying the output,
    and saving the processed video to a file.
    
    Parameters:
    - input_video_path: Path to the input video file.
    - output_video_path: Path to save the processed video file.
    """
    output_video_path = input_video_path.replace('.mp4', '_processed.mp4')

    # Open the video capture
    cap = cv2.VideoCapture(input_video_path)
    
    # Get the video frame width, height, and frame rate from the original video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Process the frame
        processed_frame = process_frame(frame)
        
        # Write the processed frame to the output video
        out.write(processed_frame)
    
    # Release video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def process_image(input_image_path):
    """
    Processes a single image, applying the shape detection and displaying the output.
    
    Parameters:
    - input_image_path: Path to the input image file.
    - area_threshold: Minimum area threshold to filter out small contours.
    """
    # Read the image
    frame = cv2.imread(input_image_path)
    # Process the image
    processed_frame = process_frame(frame)
    # Display the processed image
    cv2.imshow('Processed Image', processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Set debug mode to False
debug_mode = False

# Process a single image
# process_image('PennAir 2024 App Static.png')
# Process a video
process_video('PennAir 2024 App Dynamic.mp4')
process_video('PennAir 2024 App Dynamic Hard.mp4')