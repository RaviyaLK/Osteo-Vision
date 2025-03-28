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
