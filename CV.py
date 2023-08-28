#!/usr/bin/env python
# coding: utf-8

# # Computer Vision finetuning (using food category recognition from Food 101)

# ## Summary of methodology for this use case:
# * Assuming real-time detection is needed, used R-CNN architecture with the VGG16 model as a feature extractor. 
# * Additional layers such as a dense layer of 256 nodes and a dropout layer (with 0.5 rate to avoid overfitting) were incorporated
# * Final layer has 11 nodes that represent 11 unique food items in the initial dataset
# * Code runs through theoretical fine-tuning with domain-specific data on 11 categories
# * This approach enables processsing of input from a live camera input stream for results in a dashboard in near real-time with high latency, with a delay not exceeding 20 seconds
# * Included the model saved as a checkpoint at the end of the file for loading it on any future datasets

# #### Table of Contents: 
# * CV model code and creation
# * Steps to load the model saved from a checkpoint

# ## CV model code and creation

# In[9]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# In[1]:


# Analysis libraries 
import cv2
import numpy as np
import time 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


# TensorFlow and Keras libraries
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[3]:


# Sample dataset extraction libraries - proxy for fast food videoframes to train model 
import os 
import tarfile 
import urllib.request 


# In[4]:


# Downloading and extracting the dataset - this assumes the subdirectory 'food-101'
URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
DATA_PATH = "Food-101_dataset"

if not os.path.exists(DATA_PATH):
    print("Downloading Food-101 dataset...")
    urllib.request.urlretrieve(URL, "food-101.tar.gz")
    print("Extracting the dataset...")
    with tarfile.open("food-101.tar.gz", "r:gz") as tar:
        tar.extractall(path=DATA_PATH)


# In[ ]:


# Split the data into training and validation sets - this assumes the subdirectory 'food-101' if diff version throws error to adjust
potential_path1 = os.path.join(DATA_PATH, "food-101", "images")
potential_path2 = os.path.join(DATA_PATH, "images")

if os.path.exists(potential_path1):
    data_dir = potential_path1
elif os.path.exists(potential_path2):
    data_dir = potential_path2
else:
    raise ValueError("Could not locate the images directory in this version of the extracted dataset, check to see if diff version is hosted")
all_images = [] # assumes all image files are in the 'images' directory
for subdir in os.listdir(data_dir):
    subdir_path = os.path.join(data_dir, subdir)
    if os.path.isdir(subdir_path):
        for image_file in os.listdir(subdir_path):
            all_images.append(os.path.join(subdir_path, image_file))


# In[ ]:


# split data
train_data, test_data = train_test_split(all_images, test_size=0.2)


# In[ ]:


# Using VGG16 as a feature extractor for R-CNN
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[ ]:


# Base_model was VGG16, creating a more specific model for fine-tuning. 
# This creates some dense layers for fine-tuning.

# Flatten the output of base_model
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(11, activation='softmax')(x)  # Assuming 11 classes here - will have to be the same below


# In[ ]:


# Create the final model
model = Model(inputs=base_model.input, outputs=x)


# In[ ]:


# Fine-tuning with data on the new categoreis
# Assuming frames captured and manual labeling - if not this could be done with LLM
# Parameter rationale: 
# The adam optimizer maintains an adaptive learning rate for each parameter and can fine-tune the model more efficiently for quicker convergence
# The categorical crossentropy loss works well for multi-class classification, since that is the case for limited fast food items - penalizes the model more when its far off and less when it's close
RESTAURANT_DATA_PATH = 'path_to_restaurant_data'
if os.path.exists(RESTAURANT_DATA_PATH):
    restaurant_data = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator = restaurant_data.flow_from_directory(RESTAURANT_DATA_PATH, target_size=(224, 224))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, epochs=10)
else:
    print(f"Error: Path '{RESTAURANT_DATA_PATH}' does not exist.")


# In[ ]:


# Creates a dictionary to contain counts of predicted items
def extract_items_from_predictions(predictions):
    class_indices = np.argmax(predictions, axis=1)
    class_counts = np.bincount(class_indices, minlength=11) 
    class_names = ["class1", "class2", "class3", "class4", "class5", "class6", "class7", "class8", "class9", "class10", "class11"]
    detected_items = dict(zip(class_names, class_counts))
    return detected_items


# In[ ]:


# This class displays counts of predicted items in dashboard and shows warning if latency of updates is over 20 seconds
class Dashboard:
    def __init__(self):
        self.item_counts = {
            "class1": 0, "class2": 0, "class3": 0, "class4": 0,
            "class5": 0, "class6": 0, "class7": 0, "class8": 0,
            "class9": 0, "class10": 0, "class11": 0
        }

    def update(self, detected_items):
        start_time = time.time() # begin latency check
        for item, count in detected_items.items():
            self.item_counts[item] += count
        self.display()
        
        elapsed_time = time.time() - start_time # end latency check
        if elapsed_time > 20:
            print(f)

    def display(self):
        for item, count in self.item_counts.items():
            print(f"Warning: Dashboard update took {elapsed_time:.2f} seconds, exceeding the 20-second threshold.")
        print("-" * 50)


# In[ ]:


# Define the detection function for items
def detect_items(frame):
    # Preprocess frame, validate size and pass through the model
    frame = cv2.resize(frame, (224,224))
    processed_frame = preprocess_input(np.array([frame]))
    predictions = model.predict(processed_frame)
    
    # Extract items and their bounding boxes from predictions
    detected_items = extract_items_from_predictions(predictions)
    
    return detected_items


# In[ ]:


# Process video stream from each camera and update dashboard  
def process_camera_stream(camera_stream, dashboard, max_frames=580608000): # max_frames limits to 40 cameras in 1 week at 24 fps, prevents any massive dumps that would break system
    frame_count = 0
    while frame_count < max_frames:
        ret, frame = camera_stream.read()
        if not ret:
            break
        
        detected_items = detect_items(frame)
        
        # Update dashboard with detected items
        dashboard.update(detected_items)
        frame_count += 1


# In[ ]:


# Initialize camera streams and dashboard - camera range is hardcoded based on requirements, can be changed
camera_streams = [cv2.VideoCapture(camera_id) for camera_id in range(4*10)]

dashboard = Dashboard()  


# In[ ]:


for camera_stream in camera_streams:
    if camera_stream.isOpened():
        process_camera_stream(camera_stream, dashboard)
    else:
        print(f"Error: Couldn't open camera with ID {camera_streams.index(camera_stream)}")


    camera_stream.release()


# In[ ]:


# Save the model to a checkpoint - see below for loading
model.save('model_checkpoint.h5')


# ## Steps to load the model saved from a checkpoint

# In[ ]:


# Load the saved model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
model_path = 'model_checkpoint.h5'
loaded_model = load_model(model_path)


# In[ ]:


# Preprocess video frame data w loop to process frame by frame
video_path = 'path_to_videoframes'
cap = cv2.VideoCapture(video_path)

if cap.isOpened(): # check if video opened successfully 
    while cap.isOpened():
        ret, frame = cap.read()  # Read one frame
        if not ret:
            # If frame is not read properly, break the loop
            break

        # Resize the frame to fit the input size of the model - constraints of model
        frame = cv2.resize(frame, (224, 224))

        # Use preprocess_input from VGG16 module
        processed_frame = preprocess_input(np.array([frame]))  # Expanding dimensions

        # Pass the processed data through the loaded model to get predictions
        predictions = loaded_model.predict(processed_frame)

        # Get the class with the highest probability
        predicted_class = np.argmax(predictions, axis=1)

        # Using the model with alignment to the 11 classes above
        class_names = ["class1", "class2", "class3","class4","class5","class6","class7","class8","class9","class10","class11"]  # List all your class names in order
        print(f"Predicted Class: {class_names[predicted_class[0]]}")
    
# Release the video capture and destroy all windows if exist
cap.release()
cv2.destroyAllWindows()  
else:
    print(f"Error: Couldn't open video '{video_path}'")

