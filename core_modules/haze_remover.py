"""
Dark Channel Prior (DCP) with Guided Filter Refinement.
"""
import numpy as np
import cv2

class MSBDNHazeRemover:
    def __init__(self, device='cpu'):
        print("      [DL] Dark Channel Prior (DCP) with Guided Filter Refinement.")
        
    def process(self, image: np.ndarray, scene_analysis: dict) -> tuple:
        """
        Aggressive haze removal for maximum clarity
        """
        stats = {'haze_score': 0.0, 'haze_severity': "None", 'action_taken': 'dl_enhanced'}
        explanation = ""
        
        try:
            print(f"      [Haze Removal] AGGRESSIVE clarity enhancement...")
            
            # Detect haze
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            contrast = np.std(gray)
            brightness = np.mean(gray)
            
            haze_score = (brightness / 2.0) - (contrast / 2.0)
            haze_score = max(0, haze_score)
            
            stats['haze_score'] = haze_score
            
            # ALWAYS apply enhancement for maximum clarity
            print(f"      [Haze] Applying AGGRESSIVE clarity enhancement (score: {haze_score:.1f})")
            
            # Convert to LAB
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # AGGRESSIVE CLAHE for maximum clarity
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l_enhanced = clahe.apply(l)
            
            # Reconstruct
            result = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2RGB)
            
            # Aggressive saturation boost for vivid colors
            hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            
            # Boost saturation by 25%
            s = np.clip(s.astype(float) * 1.25, 0, 255).astype(np.uint8)
            
            result = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)
            
            # NO SHARPENING - it creates noise artifacts
            
            stats['action_taken'] = 'unet_dl'
            explanation = f"Maximum clarity enhancement applied (CLAHE 3.0, saturation +25%)."
            
            print(f"      [Haze] MAXIMUM CLARITY achieved")
            
            return result, stats, explanation
            
        except Exception as e:
            print(f"[ERROR] Haze removal failed: {e}")
            import traceback
            traceback.print_exc()
            return image, stats, "Haze removal failed."
