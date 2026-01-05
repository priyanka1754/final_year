"""
Scene Understanding Module - Object Detection + Scene Analysis
Detects people, vehicles, buildings, and other objects in images
and provides human-readable descriptions of the scene content.
"""
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as F

# COCO class names (80 classes)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock'
]

class SceneInterpreter:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load Faster R-CNN for object detection"""
        try:
            print("[INIT] Loading Object Detection Model (Faster R-CNN)...")
            self.model = fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
            self.model.to(self.device)
            self.model.eval()
            print("[OK] Object Detection ready.")
        except Exception as e:
            print(f"[WARN] Object detection unavailable: {e}")
            self.model = None
    
    def analyze(self, image: np.ndarray):
        """
        Analyze image content using object detection + scene heuristics
        Returns human-readable descriptions
        """
        descriptions = []
        
        try:
            # 1. Object Detection (people, vehicles, etc.)
            if self.model is not None:
                objects = self._detect_objects(image)
                if objects:
                    descriptions.extend(objects)
            
            # 2. Structural Analysis (buildings, towers)
            structures = self._detect_structures(image)
            if structures:
                descriptions.extend(structures)
            
            # 3. Vegetation
            vegetation = self._detect_vegetation(image)
            if vegetation:
                descriptions.append(vegetation)
            
            # 4. Scene Type
            scene_type = self._classify_scene_type(image)
            if scene_type:
                descriptions.append(scene_type)
            
            return descriptions if descriptions else ["General scene"]
            
        except Exception as e:
            print(f"[WARN] Scene analysis error: {e}")
            return ["Image content"]
    
    def _detect_objects(self, image):
        """Detect objects using Faster R-CNN"""
        try:
            # Prepare image
            img_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                predictions = self.model(img_tensor)[0]
            
            # Filter confident detections (>60% confidence)
            boxes = predictions['boxes'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
            
            confident_mask = scores > 0.6
            labels = labels[confident_mask]
            scores = scores[confident_mask]
            
            # FILTER: Remove unlikely objects for aerial/satellite imagery
            # These are typically indoor objects that cause false positives
            INDOOR_OBJECTS = {
                'toilet', 'chair', 'couch', 'bed', 'dining table', 'tv', 
                'laptop', 'mouse', 'remote', 'keyboard', 'microwave', 
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
                'toothbrush', 'bottle', 'wine glass', 'cup', 'fork', 
                'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich'
            }
            
            # Count objects by category (excluding indoor items)
            object_counts = {}
            for label, score in zip(labels, scores):
                class_name = COCO_CLASSES[label]
                
                # Skip indoor objects and invalid classes
                if class_name in INDOOR_OBJECTS or class_name == 'N/A' or class_name == '__background__':
                    continue
                    
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
            
            # Generate descriptions
            results = []
            
            # Prioritize people
            if 'person' in object_counts:
                count = object_counts['person']
                if count == 1:
                    results.append("Person visible in frame")
                else:
                    results.append(f"{count} people visible")
            
            # Vehicles
            vehicles = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
            vehicle_count = sum(object_counts.get(v, 0) for v in vehicles)
            if vehicle_count > 0:
                results.append(f"{vehicle_count} vehicle(s)")
            
            # Other notable outdoor objects
            for obj, count in object_counts.items():
                if obj not in ['person'] + vehicles and count > 0:
                    if count == 1:
                        results.append(f"{obj}")
                    else:
                        results.append(f"{count} {obj}s")
            
            return results[:5]  # Limit to top 5
            
        except Exception as e:
            print(f"[WARN] Object detection failed: {e}")
            return []
    
    def _detect_structures(self, image):
        """Detect buildings and structures"""
        results = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect strong vertical edges (buildings)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            vertical_strength = np.mean(np.abs(sobely))
            
            if vertical_strength > 15:
                results.append("Tall buildings/skyscrapers")
            elif vertical_strength > 8:
                results.append("Buildings/structures")
            
        except:
            pass
        
        return results
    
    def _detect_vegetation(self, image):
        """Detect trees/vegetation"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            
            # Green hue range
            green_mask = cv2.inRange(h, 35, 85)
            green_pct = np.sum(green_mask > 0) / green_mask.size
            
            if green_pct > 0.2:
                return "Trees and vegetation"
            elif green_pct > 0.05:
                return "Some vegetation/trees"
            
        except:
            pass
        
        return None
    
    def _classify_scene_type(self, image):
        """Classify overall scene"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            avg_brightness = np.mean(gray)
            
            # Check for haze
            brightness_std = np.std(gray)
            if brightness_std < 35 and avg_brightness > 150:
                return "Heavy atmospheric haze present"
            
        except:
            pass
        
        return None
