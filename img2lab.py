import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

input_folder = '/home/a16/zyx/TransUNet-main/datasets/generated_images/'
output_folder = '/home/a16/zyx/TransUNet-main/datasets/output_labels/'
batch_size = 1000

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a list of all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

# Process images in batches
for i in range(0, len(image_files), batch_size):
    batch_images = image_files[i:i+batch_size]
    print(f'Processing batch {i//batch_size + 1}/{len(image_files)//batch_size}')

    for image_file in batch_images:
        # Load the input image
        image = cv2.imread(os.path.join(input_folder, image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Flatten the image to a 2D array of pixels
        pixels = image.reshape(-1, 3)

        # Perform K-means clustering to group pixels
        kmeans = KMeans(n_clusters=3, random_state=0).fit(pixels)
        labels = kmeans.labels_

        # Reshape the labels to the original image shape
        label_map = labels.reshape(image.shape[:2])

        # Save the generated label map
        output_file = os.path.join(output_folder, image_file.replace('.jpg', '_label.png'))
        cv2.imwrite(output_file, label_map.astype(np.uint8))

    print(f'Batch {i//batch_size + 1} processing complete.')