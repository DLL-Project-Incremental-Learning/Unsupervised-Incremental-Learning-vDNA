# this script is an equivalent of bash file download_2d_prespective.sh
# it downloads the 2D perspective images from the KITTI-360 dataset
# and extracts the images from the zip files

# Additional information (This part is implemented based on Julia's e-mail):
# This scripts retains every 5th image in the data_rect folder and deletes the rest
# This is done to reduce the number of images in the dataset
# The total number of images in the dataset is printed at the end

import os
import zipfile
import urllib.request

train_list = [
    "2013_05_28_drive_0000_sync",
    "2013_05_28_drive_0002_sync",
    "2013_05_28_drive_0003_sync",
    "2013_05_28_drive_0004_sync",
    "2013_05_28_drive_0005_sync",
    "2013_05_28_drive_0006_sync",
    "2013_05_28_drive_0007_sync",
    "2013_05_28_drive_0009_sync",
    "2013_05_28_drive_0010_sync"
]

cam_list = ["00", "01"]

root_dir = "datasets/data/KITTI-360"
data_2d_dir = "data_2d_raw"

os.makedirs(os.path.join(root_dir, data_2d_dir), exist_ok=True)

# Change the current working directory to root_dir
os.chdir(root_dir)

total_images = 0

# Perspective images
for sequence in train_list:
    for camera in cam_list:
        zip_file = f"{sequence}_image_{camera}.zip"
        url = f"https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/data_2d_raw/{zip_file}"
        local_zip_path = os.path.join(data_2d_dir, zip_file)
        
        # Download the zip file
        urllib.request.urlretrieve(url, local_zip_path)
        
        # Unzip the file
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_2d_dir)
        
        # Remove the zip file
        os.remove(local_zip_path)
        
        # Remove all images except every 5th image
        image_dir = os.path.join(data_2d_dir, sequence, f"image_{camera}", "data_rect")
        if os.path.exists(image_dir):
            images = sorted(os.listdir(image_dir))
            for i, image in enumerate(images):
                
                if i % 5 != 0:
                    os.remove(os.path.join(image_dir, image))
            # Count remaining images in the directory
            remaining_images = len(os.listdir(image_dir))
            total_images += remaining_images
            print(f"{sequence}/image_{camera}/data_rect: {remaining_images} images")

# Print the cumulative sum
print(f"Total number of images across all folders: {total_images}")
