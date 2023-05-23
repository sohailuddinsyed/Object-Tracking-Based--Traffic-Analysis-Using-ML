import os
import json
from algorithm.object_detector import YOLOv7
import cv2




# Define the paths to the image and labels folders
image_folder = r'C:\Users\cools\Desktop\Dataset-2\images'
labels_folder = r'C:\Users\cools\Desktop\Dataset-2\labels'

# Load the list of image filenames
image_filenames = os.listdir(image_folder)

# Load the labels from the JSON files
labels = []
for filename in image_filenames:
    label_path = os.path.join(labels_folder, f"{os.path.splitext(filename)[0]}.json")
    with open(label_path, 'r') as file:
        label = json.load(file)
        labels.append(label)

# Load your trained model using the saved weights
yolov7 = YOLOv7()
yolov7.load('best.weights', classes='classes.yaml', device='cpu')

# Define variables for accuracy calculation
total_samples = len(image_filenames)
correct_predictions = 0

def load_and_preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Preprocess the image (e.g., resize, normalize, etc.)
    # Add your preprocessing steps here
    
    return image

# Iterate through the dataset and make predictions
# Iterate through the dataset and make predictions
for i in range(total_samples):
    image_filename = image_filenames[i]
    image_path = os.path.join(image_folder, image_filename)
    true_label = labels[i]
    
    # Load and preprocess the image
    image = load_and_preprocess_image(image_path)
    
   # Perform inference using the model
predictions = yolov7.detect(image, track=False)

# Print predictions for debugging
#print(predictions)

# Compare the predicted labels with the true label
true_label_found = False
for prediction in predictions:
    if isinstance(predictions, list) and len(predictions) > 0:
        prediction = predictions[0]  # Retrieve the first element from the list
        if isinstance(prediction, dict):
            if (
                str(prediction.get('class')) == str(true_label['class_id'])
                and prediction.get('x') == true_label['x']
                and prediction.get('y') == true_label['y']
                and prediction.get('width') == true_label['width']
                and prediction.get('height') == true_label['height']
                
            ):
                true_label_found = True

if true_label_found:
    correct_predictions += 1
    print(correct_predictions)


# Calculate accuracy
accuracy = correct_predictions / total_samples
print(f"Accuracy: {accuracy}")

