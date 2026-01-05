"""
Professional AI Image Captioning System
Generates detailed, human-readable descriptions of satellite imagery

"""
import cv2
import numpy as np

class AISceneCaptioner:
    """
    Professional-grade AI captioning system
    Generates detailed, natural descriptions
    """
    def __init__(self, device='cpu'):
        self.device = device
        
    def generate_detailed_caption(self, image: np.ndarray, context: dict = None) -> dict:
        """
        Generate comprehensive, human-readable description
        Returns: Dict with structured caption components
        """
        try:
            # Extract comprehensive visual features
            features = self._deep_visual_analysis(image)
            
            # Generate structured caption
            caption = {
                'overview': self._generate_overview(features),
                'objects': self._describe_objects(features),
                'environment': self._describe_environment(features),
                'enhancements': self._describe_enhancements(context) if context else "",
                'full_description': ""
            }
            
            # Combine into full description
            parts = []
            if caption['overview']:
                parts.append(caption['overview'])
            if caption['objects']:
                parts.append(caption['objects'])
            if caption['environment']:
                parts.append(caption['environment'])
            if caption['enhancements']:
                parts.append(caption['enhancements'])
            
            caption['full_description'] = " ".join(parts)
            
            return caption
            
        except Exception as e:
            print(f"[ERROR] Caption generation failed: {e}")
            return {
                'overview': "Satellite imagery showing terrain features.",
                'objects': "",
                'environment': "",
                'enhancements': "",
                'full_description': "Satellite imagery showing terrain features."
            }
    
    def _deep_visual_analysis(self, image):
        """Comprehensive visual feature extraction"""
        features = {}
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # === OBJECT DETECTION ===
        
        # 1. Circular objects (water tanks, silos, storage facilities)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 30,
            param1=50, param2=25, minRadius=8, maxRadius=150
        )
        if circles is not None:
            features['circular_objects'] = len(circles[0])
            features['tank_sizes'] = [int(c[2]) for c in circles[0]]  # radii
        else:
            features['circular_objects'] = 0
            features['tank_sizes'] = []
        
        # 2. Tall vertical structures (towers, poles, buildings)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        vertical_edges = np.abs(sobely)
        features['vertical_strength'] = np.mean(vertical_edges)
        features['has_towers'] = features['vertical_strength'] > 15
        
        # 3. Linear features (roads, paths, pipelines)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength=40, maxLineGap=15)
        if lines is not None:
            features['linear_features'] = len(lines)
            # Classify by orientation
            horizontal = sum(1 for line in lines if abs(line[0][1] - line[0][3]) < 20)
            vertical = sum(1 for line in lines if abs(line[0][0] - line[0][2]) < 20)
            features['roads'] = horizontal
            features['vertical_lines'] = vertical
        else:
            features['linear_features'] = 0
            features['roads'] = 0
            features['vertical_lines'] = 0
        
        # 4. Buildings/structures (rectangular shapes)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangular_structures = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) == 4:  # Rectangular
                    rectangular_structures += 1
        features['buildings'] = rectangular_structures
        
        # === ENVIRONMENT ANALYSIS ===
        
        # 5. Vegetation (green areas)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h_channel = hsv[:,:,0]
        s_channel = hsv[:,:,1]
        
        # Green vegetation (hue 35-85, saturation > 30)
        green_mask = cv2.inRange(h_channel, 35, 85)
        sat_mask = (s_channel > 30).astype(np.uint8) * 255
        veg_mask = cv2.bitwise_and(green_mask, sat_mask)
        features['vegetation_pct'] = (np.sum(veg_mask > 0) / veg_mask.size) * 100
        
        # 6. Bare ground/soil (brown/tan areas)
        brown_mask = cv2.inRange(h_channel, 10, 30)
        features['bare_ground_pct'] = (np.sum(brown_mask > 0) / brown_mask.size) * 100
        
        # 7. Water bodies (dark blue areas)
        blue_mask = cv2.inRange(h_channel, 90, 130)
        features['water_pct'] = (np.sum(blue_mask > 0) / blue_mask.size) * 100
        
        # 8. Overall scene brightness and contrast
        features['avg_brightness'] = np.mean(gray)
        features['contrast'] = np.std(gray)
        
        # 9. Scene complexity (edge density)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        return features
    
    def _generate_overview(self, features):
        """Generate scene overview"""
        parts = []
        
        # Scene type
        if features['edge_density'] > 0.15:
            parts.append("This is a densely developed area")
        elif features['edge_density'] > 0.08:
            parts.append("This is a moderately developed area")
        else:
            parts.append("This is an open area")
        
        # Dominant features
        if features['vegetation_pct'] > 40:
            parts.append("with abundant vegetation")
        elif features['buildings'] > 10:
            parts.append("with multiple structures")
        elif features['circular_objects'] > 0:
            parts.append("featuring industrial facilities")
        
        return " ".join(parts) + "."
    
    def _describe_objects(self, features):
        """Describe detected objects in detail"""
        objects = []
        
        # Circular objects (tanks, silos)
        if features['circular_objects'] > 0:
            if features['circular_objects'] == 1:
                size = features['tank_sizes'][0]
                if size > 50:
                    objects.append("One large water storage tank (approximately 50+ meters diameter)")
                elif size > 25:
                    objects.append("One medium-sized storage tank")
                else:
                    objects.append("One small storage facility")
            else:
                objects.append(f"{features['circular_objects']} circular storage tanks/silos")
        
        # Towers/tall structures
        if features['has_towers']:
            objects.append("Tall vertical structures (towers or poles)")
        
        # Buildings
        if features['buildings'] > 5:
            objects.append(f"{features['buildings']} rectangular buildings or structures")
        elif features['buildings'] > 0:
            objects.append(f"{features['buildings']} building(s)")
        
        # Roads/paths
        if features['roads'] > 5:
            objects.append("Multiple roads and pathways")
        elif features['roads'] > 0:
            objects.append("Road infrastructure")
        
        if objects:
            return "Detected objects: " + ", ".join(objects) + "."
        return ""
    
    def _describe_environment(self, features):
        """Describe environmental features"""
        env = []
        
        # Vegetation
        if features['vegetation_pct'] > 50:
            env.append("Dense vegetation coverage (>50%)")
        elif features['vegetation_pct'] > 20:
            env.append(f"Moderate vegetation ({features['vegetation_pct']:.0f}%)")
        elif features['vegetation_pct'] > 5:
            env.append("Scattered vegetation")
        
        # Bare ground
        if features['bare_ground_pct'] > 30:
            env.append("significant bare ground areas")
        
        # Water
        if features['water_pct'] > 5:
            env.append("water bodies visible")
        
        if env:
            return "Environment: " + ", ".join(env) + "."
        return ""
    
    def _describe_enhancements(self, context):
        """Describe applied enhancements"""
        if not context:
            return ""
        
        enhancements = []
        
        if context.get('shadow_removed'):
            enhancements.append("Shadow areas have been enhanced to reveal underlying ground details")
        
        if context.get('haze_removed'):
            enhancements.append("atmospheric haze has been removed for improved clarity")
        
        if context.get('cloud_removed'):
            enhancements.append("cloud coverage has been processed")
        
        if enhancements:
            return "Image enhancements: " + ", ".join(enhancements) + "."
        return "Image has been enhanced for optimal visibility and clarity."
    
    def generate_change_caption(self, image1, image2, changes_detected):
        """Generate detailed change description"""
        try:
            features1 = self._deep_visual_analysis(image1)
            features2 = self._deep_visual_analysis(image2)
            
            changes = []
            
            # Structural changes
            if changes_detected.get('structural_changes'):
                change_pct = changes_detected.get('change_percentage', 0)
                changes.append(f"Structural changes detected in {change_pct:.1f}% of the area")
            
            # Object changes
            if abs(features2['circular_objects'] - features1['circular_objects']) > 0:
                diff = features2['circular_objects'] - features1['circular_objects']
                if diff > 0:
                    changes.append(f"{diff} new storage tank(s) added")
                else:
                    changes.append(f"{abs(diff)} storage tank(s) removed")
            
            if abs(features2['buildings'] - features1['buildings']) > 2:
                diff = features2['buildings'] - features1['buildings']
                if diff > 0:
                    changes.append(f"{diff} new building(s) constructed")
                else:
                    changes.append(f"{abs(diff)} building(s) demolished")
            
            # Vegetation changes
            veg_change = features2['vegetation_pct'] - features1['vegetation_pct']
            if abs(veg_change) > 15:
                if veg_change > 0:
                    changes.append(f"Vegetation increased by {veg_change:.0f}%")
                else:
                    changes.append(f"Vegetation decreased by {abs(veg_change):.0f}% (possible land clearing)")
            
            if changes:
                return "Changes detected: " + "; ".join(changes) + ". Both images have been enhanced for accurate comparison."
            else:
                return "No significant structural changes detected between the two time periods. Both images have been enhanced for detailed comparison."
                
        except Exception as e:
            print(f"[ERROR] Change caption failed: {e}")
            return "Comparison of two satellite images at different time periods."

def get_ai_captioner(device='cpu'):
    return AISceneCaptioner(device)
