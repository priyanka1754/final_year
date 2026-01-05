"""
Cloud Processing - Disabled (Pure DL Mode)
Cloud removal requires specialized models not yet integrated
Detection: Luma-Thresholding (>240) with Morphological Filtering.
Removal: Telea Inpainting for small scattered coverage.
"""
import numpy as np

class CloudRemover:
    def process(self, image: np.ndarray) -> tuple:
        """
        Cloud processing disabled in pure DL mode
        Returns original image unchanged
        """
        stats = {'cloud_pct': 0.0, 'detected': False}
        
        # No classical methods - return original
        return image, stats
