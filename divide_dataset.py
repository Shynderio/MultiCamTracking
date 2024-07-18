import os

def rename_file(image_dir, extension):
    # List all files in the directory
    images = sorted(os.listdir(image_dir))
    
    # Define the renaming format and counters
    person1_cam1_counter = 1
    person2_cam1_counter = 1
    person1_cam2_counter = 1
    person2_cam2_counter = 1
    
    # Renaming images according to the given distribution
    for i, image in enumerate(images):
        if i < 9:  # First 9 images are person1 in cam1
            new_name = f'person1_cam1_{person1_cam1_counter}.{extension}'
            person1_cam1_counter += 1
        elif i < 15:  # Next 6 images are person2 in cam1
            new_name = f'person2_cam1_{person2_cam1_counter}.{extension}'
            person2_cam1_counter += 1
        elif i < 30:  # Next 18 images are person1 in cam2
            new_name = f'person1_cam2_{person1_cam2_counter}.{extension}'
            person1_cam2_counter += 1
        else:  # Remaining images are person2 in cam2
            new_name = f'person2_cam2_{person2_cam2_counter}.{extension}'
            person2_cam2_counter += 1
        
        # Full path to old and new file names
        old_file = os.path.join(image_dir, image)
        new_file = os.path.join(image_dir, new_name)
        
        # Renaming the file
        os.rename(old_file, new_file)
        print(f'Renamed: {old_file} to {new_file}')

# Example usage
image_dir = 'train-20240716T062652Z-001/train/images'  # Update this to your actual directory path
label_dir = 'train-20240716T062652Z-001/train/labels' 
rename_file(label_dir, 'txt')
rename_file(image_dir, 'jpg')
