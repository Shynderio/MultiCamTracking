import cv2
import albumentations as A
import numpy as np

# Define the augmentation pipeline
transform = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.Transpose(),
    A.OneOf([
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    A.OneOf([
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=0.1),
        A.ElasticTransform(p=0.3),  # Using ElasticTransformation instead of PiecewiseAffine
    ], p=0.2),
    A.OneOf([
        A.HueSaturationValue(p=0.3),
        A.RGBShift(p=0.1),
        A.RandomBrightnessContrast(p=0.3),
    ], p=0.2),
])

# Load video
cap = cv2.VideoCapture('data/sam.mp4')
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (frame_width, frame_height))


frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply augmentations
    augmented = transform(image=frame)
    augmented_frame = augmented['image']
    
    # Write the frame to the output video
    out.write(augmented_frame)
    
    # Save frames as images (optional, for checking results)
    cv2.imwrite(f'augmented_frames/frame_{frame_count}.jpg', augmented_frame)
    frame_count += 1

cap.release()
out.release()
