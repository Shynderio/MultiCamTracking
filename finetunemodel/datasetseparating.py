import os
import shutil

def copy_dataset(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create train, query, and gallery folders
    train_folder = os.path.join(output_folder, 'train')
    query_folder = os.path.join(output_folder, 'query')
    gallery_folder = os.path.join(output_folder, 'gallery')
    os.makedirs(train_folder)
    os.makedirs(query_folder)
    os.makedirs(gallery_folder)

    # Copy images to train folder
    for person_folder in os.listdir(input_folder):
        person_path = os.path.join(input_folder, person_folder)
        if os.path.isdir(person_path):
            train_person_folder = os.path.join(train_folder, person_folder)
            os.makedirs(train_person_folder)
            for camera_folder in os.listdir(person_path):
                camera_path = os.path.join(person_path, camera_folder)
                if os.path.isdir(camera_path):
                    train_camera_folder = os.path.join(train_person_folder, camera_folder)
                    os.makedirs(train_camera_folder)
                    for image_file in os.listdir(camera_path):
                        image_path = os.path.join(camera_path, image_file)
                        if os.path.isfile(image_path):
                            shutil.copy(image_path, train_camera_folder)

    # Copy images to query folder
    for image_file in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_file)
        if os.path.isfile(image_path):
            shutil.copy(image_path, query_folder)

    # Copy images to gallery folder
    for person_folder in os.listdir(input_folder):
        person_path = os.path.join(input_folder, person_folder)
        if os.path.isdir(person_path):
            for camera_folder in os.listdir(person_path):
                camera_path = os.path.join(person_path, camera_folder)
                if os.path.isdir(camera_path):
                    for image_file in os.listdir(camera_path):
                        image_path = os.path.join(camera_path, image_file)
                        if os.path.isfile(image_path):
                            shutil.copy(image_path, gallery_folder)

# Define input and output folders
input_folder = "input dataset path"
output_folder = "output dataset path"

# Copy dataset
copy_dataset(input_folder, output_folder)

print("Dataset copied and formatted successfully.")