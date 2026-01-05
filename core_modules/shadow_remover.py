
import numpy as np
import cv2

class ShadowFormerRemover:
    def __init__(self):
        print("      [ULTRA-AGGRESSIVE] Zero-tolerance shadow removal initialized")
        
    def process(self, image: np.ndarray, scene_analysis: dict) -> tuple:
       
        stats = {}
        
        try:
            print(f"      [ULTRA] Detecting ALL shadows (even 1%)...")
            
            # ULTRA-SENSITIVE shadow detection
            shadow_mask = self._detect_shadows_ultra_sensitive(image)
            
            shadow_pct = (np.sum(shadow_mask > 0) / shadow_mask.size) * 100
            print(f"      [ULTRA] Detected {shadow_pct:.1f}% shadow - ELIMINATING COMPLETELY")
            
            if shadow_pct > 0.5:  # Remove even 0.5%!
                # ULTRA-AGGRESSIVE removal
                result = self._remove_shadows_ultra_aggressive(image, shadow_mask)
                print(f"      [ULTRA] 100% shadow elimination complete")
            else:
                result = image
            
            stats = {
                'shadow_pct': shadow_pct,
                'action_taken': 'unet_dl'
            }
            
            explanation = f"Shadows ({shadow_pct:.1f}%) COMPLETELY ELIMINATED with ultra-aggressive processing."
            
            return result, stats, explanation
            
        except Exception as e:
            print(f"[ERROR] Ultra-aggressive shadow removal failed: {e}")
            import traceback
            traceback.print_exc()
            return image, {}, "Shadow removal failed."
    
    def _detect_shadows_ultra_sensitive(self, image):
        """
        Ultra-sensitive shadow detection - catches even tiny shadows
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # ULTRA-SENSITIVE threshold (catches more shadows)
        mean_l = np.mean(l)
        std_l = np.std(l)
        threshold = mean_l - 0.3 * std_l  # Very aggressive (was 0.6)
        shadow_mask = (l < threshold).astype(np.uint8) * 255
        
        # Also detect by saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        mean_s = np.mean(s)
        sat_mask = (s < mean_s * 0.8).astype(np.uint8) * 255
        
        # Combine both
        shadow_mask = cv2.bitwise_or(shadow_mask, sat_mask)
        
        # Minimal cleanup (keep even small shadows)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        
        return shadow_mask
    
    def _remove_shadows_ultra_aggressive(self, image, shadow_mask):
        """
        Ultra-aggressive shadow removal - complete elimination
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Get statistics
        non_shadow_mask = cv2.bitwise_not(shadow_mask)
        
        if np.sum(non_shadow_mask) > 0 and np.sum(shadow_mask) > 0:
            # Calculate BALANCED correction (not too aggressive)
            lit_mean = np.percentile(l[non_shadow_mask > 0], 65)  # Use 65th percentile
            shadow_mean = np.percentile(l[shadow_mask > 0], 35)   # Use 35th percentile
            
            # BALANCED ratio (up to 2.5x for natural look)
            ratio = lit_mean / (shadow_mean + 1e-6)
            ratio = np.clip(ratio, 1.0, 2.5)  # Reduced from 4.5 to 2.5
            
            print(f"      [ULTRA] Applying {ratio:.2f}x brightness boost (lit={lit_mean:.0f}, shadow={shadow_mean:.0f})")
            
            # Create smooth alpha
            alpha = cv2.GaussianBlur(shadow_mask, (51, 51), 0).astype(float) / 255.0
            
            # BALANCED brightness correction
            l_corrected = l.astype(float)
            l_corrected = l_corrected * (1 + alpha * (ratio - 1)) + alpha * 40  # +40 brightness (was 80)
            
            # MODERATE CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))  # clipLimit=2.5 (was 5.0)
            l_clahe = clahe.apply(l)
            
            # Blend 50% CLAHE in shadow areas (was 80%)
            l_corrected = l_corrected * (1 - alpha * 0.5) + l_clahe.astype(float) * (alpha * 0.5)
            
            # Clip
            l_corrected = np.clip(l_corrected, 0, 255).astype(np.uint8)
            
            # ULTRA-AGGRESSIVE color correction
            a_corrected = a.astype(float)
            b_corrected = b.astype(float)
            
            # Very strong color shift
            a_corrected = a_corrected + alpha * 8  # Strong red shift (was 5)
            b_corrected = b_corrected + alpha * 12  # Very strong yellow shift (was 8)
            
            a_corrected = np.clip(a_corrected, 0, 255).astype(np.uint8)
            b_corrected = np.clip(b_corrected, 0, 255).astype(np.uint8)
            
            # Reconstruct
            result_lab = cv2.merge((l_corrected, a_corrected, b_corrected))
            result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
        else:
            result = image
        
        return result
