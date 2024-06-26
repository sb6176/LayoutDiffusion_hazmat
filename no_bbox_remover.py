import json
import os
from collections import defaultdict

# Define paths
annotation_file_path = './data/annotations/instances_val.json'
images_dir_path = './data/images256/val'
output_annotation_file_path = './data/annotations/filtered_instances_val.json'

# Load COCO annotations
with open(annotation_file_path, 'r') as f:
    coco_data = json.load(f)

# Get lists of annotations and images
annotations = coco_data['annotations']
images = coco_data['images']

# Create a dictionary to hold annotations by image_id
annotations_by_image = defaultdict(list)

for annotation in annotations:
    image_id = annotation['image_id']
    annotations_by_image[image_id].append(annotation)

# Identify images with no bounding boxes
images_to_keep = []
annotations_to_keep = []

for image in images:
    image_id = image['id']
    if image_id in annotations_by_image and annotations_by_image[image_id]:
        images_to_keep.append(image)
        annotations_to_keep.extend(annotations_by_image[image_id])
    else:
        # Remove corresponding image file
        image_path = os.path.join(images_dir_path, image['file_name'])
        if os.path.exists(image_path):
            print(f"Removing image file: {image_path}")
            os.remove(image_path)

# Update the COCO data structure with filtered images and annotations
filtered_coco_data = coco_data.copy()
filtered_coco_data['images'] = images_to_keep
filtered_coco_data['annotations'] = annotations_to_keep

# Save the updated annotations to a new file
with open(output_annotation_file_path, 'w') as f:
    json.dump(filtered_coco_data, f, indent=2)

print(f"Filtered annotations saved to {output_annotation_file_path}")
print(f"Number of remaining images: {len(images_to_keep)}")
print(f"Number of remaining annotations: {len(annotations_to_keep)}")
