import os
import shutil

# Function to create folder structure and copy images
def organize_gallery(gallery_folder):
    # Create folders for each person ID
    for person_id in range(1, 10):
        person_folder = os.path.join(gallery_folder, f'person{person_id}')
        os.makedirs(person_folder, exist_ok=True)
        
        # Create folders for each camera ID
        for camera_id in range(1, 6):
            camera_folder = os.path.join(person_folder, f'camera{camera_id}')
            os.makedirs(camera_folder, exist_ok=True)
    
    # Copy images to respective folders based on filename
    for filename in os.listdir(gallery_folder):
        if filename.endswith('.jpg'):
            parts = filename.split('_')
            person_id = int(parts[0])
            camera_id = int(parts[1])
            image_index = parts[2].split('.')[0]
            source_path = os.path.join(gallery_folder, filename)
            destination_folder = os.path.join(gallery_folder, f'person{person_id}', f'camera{camera_id}')
            destination_path = os.path.join(destination_folder, filename)
            shutil.copy(source_path, destination_path)

# Specify the path to the gallery folder
gallery_folder = 'the divided gallery folder path'

# Organize the gallery images
organize_gallery(gallery_folder)

print("Gallery images organized successfully.")