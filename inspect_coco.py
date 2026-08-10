import json
import os

with open(r"c:\Users\prajn\OneDrive\Desktop\iitp\news.coco\train\_annotations.coco.json") as f:
    coco = json.loads(f.read())

images = coco.get("images", [])
annotations = coco.get("annotations", [])
categories = coco.get("categories", [])

print("Categories:", categories)

print("Number of images in JSON:", len(images))
print("Number of annotations:", len(annotations))

image_dict = {img["id"]: img for img in images}
filenames = [img["file_name"] for img in images]
print("Duplicate filenames:", len(filenames) - len(set(filenames)))

missing_images = []
for fname in filenames:
    if not os.path.exists(os.path.join(r"c:\Users\prajn\OneDrive\Desktop\iitp\news.coco\train", fname)):
        missing_images.append(fname)
print("Missing images from disk:", len(missing_images))

missing_img_refs = []
out_of_bounds = 0
invalid_boxes = 0
widths = []
heights = []
zero_ann_images = set(img["id"] for img in images)
dims = set((img.get("width"), img.get("height")) for img in images)

for ann in annotations:
    img_id = ann["image_id"]
    if img_id in zero_ann_images:
        zero_ann_images.remove(img_id)
        
    if img_id not in image_dict:
        missing_img_refs.append(img_id)
        continue
        
    img = image_dict[img_id]
    bbox = ann["bbox"] # [x, y, width, height]
    if len(bbox) != 4 or bbox[2] < 0 or bbox[3] < 0:
        invalid_boxes += 1
    else:
        widths.append(bbox[2])
        heights.append(bbox[3])
        x, y, w, h = bbox
        if x < 0 or y < 0 or x + w > img["width"] or y + h > img["height"]:
            out_of_bounds += 1

print("Missing image refs in annotations:", len(missing_img_refs))
print("Valid boxes format [x,y,w,h]:", invalid_boxes == 0)
print("Consistent dimensions:", len(dims) == 1, dims)
print("Zero annotation images:", len(zero_ann_images))

if widths:
    print(f"Min w: {min(widths)}, Max w: {max(widths)}, Avg w: {sum(widths)/len(widths):.2f}")
    print(f"Min h: {min(heights)}, Max h: {max(heights)}, Avg h: {sum(heights)/len(heights):.2f}")
print("Out of bounds boxes:", out_of_bounds)
