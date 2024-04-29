import os
import shutil
import subprocess
import tarfile
import os
import pandas as pd
import numpy as np
from PIL import Image

print("Preparing dataset..")
with tarfile.open("fer2013.tar.gz", "r") as tar:
    tar.extractall("fer2013")

output_folder_path = "FER2013"

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv("fer2013/fer2013/fer2013.csv")

# Define a dictionary to map emotion codes to labels
emotion_labels = {
    "0": "Angry",
    "1": "Disgust",
    "2": "Fear",
    "3": "Happy",
    "4": "Sad",
    "5": "Surprise",
    "6": "Neutral",
}

# Create the output folders and subfolders if they do not exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
for usage in ["train", "val", "test"]:
    usage_folder_path = os.path.join(output_folder_path, usage)
    if not os.path.exists(usage_folder_path):
        os.makedirs(usage_folder_path)
    for label in emotion_labels.values():
        subfolder_path = os.path.join(usage_folder_path, label)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

# Loop over each row in the DataFrame
for index, row in df.iterrows():
    # Extract the image data from the row
    pixels = row["pixels"].split()
    img_data = [int(pixel) for pixel in pixels]
    img_array = np.array(img_data).reshape(48, 48)
    img = Image.fromarray(img_array.astype("uint8"), "L")

    # Get the emotion label and determine the output subfolder based on the Usage column
    emotion_label = emotion_labels[str(row["emotion"])]
    if row["Usage"] == "Training":
        output_subfolder_path = os.path.join(output_folder_path, "train", emotion_label)
    elif row["Usage"] == "PublicTest":
        output_subfolder_path = os.path.join(output_folder_path, "val", emotion_label)
    else:
        output_subfolder_path = os.path.join(output_folder_path, "test", emotion_label)

    # Save the image to the output subfolder
    output_file_path = os.path.join(output_subfolder_path, f"{index}.jpg")
    img.save(output_file_path)
