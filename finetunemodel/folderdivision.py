import os
import shutil

# Define the dataset directory
dataset_dir = "path of dataset"

# Define the identities present in each camera
# This camera identities may vary based on your dataset
camera_identities = {
    1: [1, 2, 3, 4, 5, 6],
    2: [3, 5, 6, 7],
    3: [3, 5, 6],
    4: [3, 5, 6, 8, 9],
    5: [3, 5, 6, 7, 8]
}

# Define a function to assign names to images
def assign_names(person_dir, person_id, camera_id):
    image_files = os.listdir(person_dir)
    for i, image_file in enumerate(image_files, start=1):
        src_path = os.path.join(person_dir, image_file)
        filename, extension = os.path.splitext(image_file)
        dst_path = os.path.join(person_dir, f"{person_id}_{camera_id}_{i:03d}{extension}")
        shutil.move(src_path, dst_path)

# Traverse the dataset directory
for person_id, person_dir in enumerate(sorted(os.listdir(dataset_dir)), start=1):
    camera_path = os.path.join(dataset_dir, camera_dir)
    identities = camera_identities[camera_id]
    for person_id in identities:
        person_dir = os.path.join(camera_path, str(person_id))
        if os.path.exists(person_dir):
            assign_names(person_dir, person_id, camera_id)