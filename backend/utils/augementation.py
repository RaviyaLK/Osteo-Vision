# import os
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
# import numpy as np
# 
# # === SET YOUR PATHS HERE ===
# input_dir = r"C:\Users\Metropolitan\OneDrive\Desktop\Osteo-Vision\Data\Train\Osteopenia"  # Original osteopenia images
# output_dir = r"C:\Users\Metropolitan\OneDrive\Desktop\Osteo-Vision\Data\Train\Augmented_2"  # Save augmented images here
# 
# # Create the output folder if not exists
# os.makedirs(output_dir, exist_ok=True)
# 
# # Setup the augmentation generator
# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.15,
#     zoom_range=0.1,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )
# 
# # Load all original images
# image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
# total_to_generate = 60
# generated_count = 0
# 
# for image_file in image_files:
#     if generated_count >= total_to_generate:
#         break
# 
#     image_path = os.path.join(input_dir, image_file)
#     img = load_img(image_path)  # Load image
#     x = img_to_array(img)       # Convert to array
#     x = np.expand_dims(x, axis=0)  # Reshape for generator
# 
#     # Generate one augmentation from this image
#     aug_iter = datagen.flow(x, batch_size=1)
# 
#     for _ in range(1):  # Only one augmentation per image
#         aug_img = next(aug_iter)[0].astype('uint8')
#         aug_img = array_to_img(aug_img)
# 
#         new_filename = f"osteopenia_aug_extra_{generated_count+1}.jpg"
#         aug_img.save(os.path.join(output_dir, new_filename))
#         generated_count += 1
# 
#         if generated_count >= total_to_generate:
#             break
# 
# print(f"✅ Successfully generated {generated_count} additional augmented images.")
import cv2
import os

# Directory where your images are located
input_directory = r"C:\Users\Metropolitan\OneDrive\Desktop\Osteo-Vision\DatasetUltra\Osteopenia"
# Directory where you want to save the modified images
output_directory =  r"C:\Users\Metropolitan\OneDrive\Desktop\Osteo-Vision\DatasetUltra\New"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Loop through the images in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".jpg"):  # Modify the file extension as needed
        # Load the image
        image_path = os.path.join(input_directory, filename)
        img = cv2.imread(image_path)

        # Zoom the image (resize)
        zoomed_img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Flip the image horizontally
        flipped_img = cv2.flip(zoomed_img, 1)

        # Save the modified image
        output_path = os.path.join(output_directory, filename)
        cv2.imwrite(output_path, flipped_img)

print("Images have been processed and saved to the output directory.")
