import cv2
import numpy as np
from core_modules.haze_remover import MSBDNHazeRemover
from core_modules.shadow_remover import ShadowFormerRemover
from core_modules.cloud_remover import CloudRemover
from core_modules.scene_interpreter import SceneInterpreter
from core_modules.scene_analyzer import SceneAnalyzer
from core_modules.ai_captioner import AISceneCaptioner

class DeepLearningEnhancer:
    def __init__(self, device='cpu'):
        print("[INIT] Loading AI-Powered Satellite Vision System...")
        print("  - Deep Learning Shadow Removal")
        print("  - Deep Learning Haze Removal")
        print("  - Scene Understanding")
        print("  - Natural Language Captioning (BLIP-2 style)")
        self.scene_analyzer = SceneAnalyzer()
        self.haze_remover = MSBDNHazeRemover(device)
        self.shadow_remover = ShadowFormerRemover()
        self.cloud_remover = CloudRemover()
        self.scene_interpreter = SceneInterpreter(device)
        self.ai_captioner = AISceneCaptioner(device)

    def enhance(self, image: np.ndarray, tasks: list = ['all']) -> dict:
        """
        Semantic understanding pipeline
        PRIORITIZES INTERPRETABILITY AND CLARITY
        - Removes ALL shadows for semantic understanding
        - Enhances all images for better object recognition
        - Generates natural language explanations
        """
        # DEFENSE: Ensure input is valid
        if image is None or not isinstance(image, np.ndarray):
            print("[CRITICAL] Input image is None/Invalid!")
            return self._create_safe_output(image)

        original = image.copy()
        result = image.copy()
        applied = []
        detection_raw = {}
        explanations = []
        
        # ═══════════════════════════════════════════════════════════
        # STEP 0: MANDATORY SCENE UNDERSTANDING
        # ═══════════════════════════════════════════════════════════
        print("   [STEP 0] Scene Understanding (Mandatory)...")
        scene_analysis = self.scene_analyzer.analyze(image)
        
        print(f"      Scene Type: {scene_analysis['scene_type']}")
        print(f"      Tall Structures: {scene_analysis['has_tall_structures']}")
        if scene_analysis['shadow_geometry'] is not None:
            print(f"      Shadow Geometry: {scene_analysis['shadow_geometry']['type']}")
        else:
            print(f"      Shadow Geometry: N/A (disabled in pure DL mode)")
        
        # Semantic content analysis
        print("   [ANALYSIS] Interpreting Scene Content...")
        scene_tags = self.scene_interpreter.analyze(image)
        detection_raw['content'] = scene_tags
        
        try:
            # ═══════════════════════════════════════════════════════════
            # STEP 1: SHADOW HANDLING (Physics-Based)
            # ═══════════════════════════════════════════════════════════
            if 'shadow' in tasks or 'all' in tasks:
                result, s_stats, s_explanation = self.shadow_remover.process(result, scene_analysis)
                detection_raw['shadow'] = s_stats
                explanations.append(s_explanation)
                
                # Count any DL enhancement as applied
                if s_stats['action_taken'] in ['corrected', 'unet_dl', 'resnet18_fast', 'swin2sr_transformer']:
                    applied.append('shadow_removal_dl')
                elif s_stats['action_taken'] == 'preserved':
                    applied.append('shadow_preserved')

            # ═══════════════════════════════════════════════════════════
            # STEP 2: HAZE REMOVAL (Confidence-Gated)
            # ═══════════════════════════════════════════════════════════
            if 'haze' in tasks or 'all' in tasks:
                result, h_stats, h_explanation = self.haze_remover.process(result, scene_analysis)
                detection_raw['haze'] = h_stats
                explanations.append(h_explanation)
                
                # Count any DL enhancement as applied
                if h_stats['action_taken'] in ['corrected', 'unet_dl', 'resnet18_fast', 'swin2sr_transformer']:
                    applied.append('haze_removal_dl')
            
            # ═══════════════════════════════════════════════════════════
            # STEP 3: CLOUD HANDLING (No Hallucination)
            # ═══════════════════════════════════════════════════════════
            if 'cloud' in tasks or 'all' in tasks:
                result, c_stats = self.cloud_remover.process(result)
                detection_raw['cloud'] = c_stats
                
                if c_stats['detected'] and c_stats['cloud_pct'] < 10:
                    applied.append('cloud_removal_scattered')
                    explanations.append(f"Scattered clouds ({c_stats['cloud_pct']:.1f}%) gently inpainted.")
                elif c_stats['detected']:
                    explanations.append(f"Dense clouds ({c_stats['cloud_pct']:.1f}%) detected - single-image removal not possible. Original preserved.")

            # ═══════════════════════════════════════════════════════════
            # AI DETAILED CAPTION GENERATION
            # ═══════════════════════════════════════════════════════════
            caption_context = {
                'shadow_removed': 'shadow_removal' in applied or 'shadow_correction' in applied,
                'haze_removed': 'haze_removal' in applied,
                'cloud_removed': 'cloud_removal_scattered' in applied,
                'detection': detection_raw
            }
            
            detailed_caption = self.ai_captioner.generate_detailed_caption(result, caption_context)
            
            # Clear previous explanations and use AI-generated detailed description
            explanations = [
                f"📋 SCENE ANALYSIS:",
                f"   {detailed_caption['overview']}",
                f"   {detailed_caption['objects']}" if detailed_caption['objects'] else "",
                f"   {detailed_caption['environment']}" if detailed_caption['environment'] else "",
                "",
                f"✨ ENHANCEMENTS APPLIED:",
                f"   {detailed_caption['enhancements']}" if detailed_caption['enhancements'] else "   Image optimized for maximum visibility."
            ]
            
            # Remove empty strings
            explanations = [e for e in explanations if e]
            
            # ═══════════════════════════════════════════════════════════
            # PURE DEEP LEARNING - NO CLASSICAL POST-PROCESSING
            # ═══════════════════════════════════════════════════════════

            # DEFENSE: Final type check
            if result.dtype != np.uint8:
                result = np.clip(result, 0, 255).astype(np.uint8)

            return {
                'processed': result,
                'detection': self._format_detection(detection_raw, scene_tags),
                'applied_corrections': applied,
                'explanations': explanations,
                'scene_analysis': scene_analysis
            }

        except Exception as e:
            print(f"[ERROR] Enhancement failed: {e}")
            import traceback
            traceback.print_exc()
            return self._create_safe_output(original)
    
    
    def _format_detection(self, raw, scene_tags):
        """Format detection results"""
        return {
            'content': scene_tags,
            'shadow': {
                'detected': raw.get('shadow', {}).get('shadow_severity', 'None') != 'None',
                'percentage': raw.get('shadow', {}).get('shadow_pct', 0),
                'severity': raw.get('shadow', {}).get('shadow_severity', 'None'),
                'action': raw.get('shadow', {}).get('action_taken', 'none')
            },
            'haze': {
                'detected': raw.get('haze', {}).get('haze_severity', 'None') != 'None',
                'score': raw.get('haze', {}).get('haze_score', 0),
                'severity': raw.get('haze', {}).get('haze_severity', 'None'),
                'confidence': raw.get('haze', {}).get('confidence', 0),
                'action': raw.get('haze', {}).get('action_taken', 'none')
            },
            'cloud': {
                'detected': raw.get('cloud', {}).get('detected', False),
                'percentage': raw.get('cloud', {}).get('cloud_pct', 0)
            }
        }
    
    def _create_safe_output(self, image):
        """Create safe fallback output"""
        if image is None:
            image = np.zeros((256,256,3), dtype=np.uint8)
        return {
            'processed': image,
            'detection': {},
            'applied_corrections': ['error_fallback'],
            'explanations': ['Processing failed - original preserved'],
            'scene_analysis': {}
        }


def get_dl_enhancer(device='cpu'):
    """Factory function"""
    return DeepLearningEnhancer(device)
