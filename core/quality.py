import cv2
import numpy as np
from config import settings

class FaceQualityFilter:
    def __init__(self, min_blur_var: float = settings.MIN_BLUR_VAR, min_size: int = settings.MIN_FACE_SIZE):
        self.min_blur_var = min_blur_var
        self.min_size = min_size

    def evaluate_quality(self, crop_bgr: np.ndarray) -> tuple[bool, float, str]:
        """
        Evaluates the quality of a face crop.
        Returns (is_passed, blur_score, reason)
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return False, 0.0, "Empty crop"

        h, w = crop_bgr.shape[:2]
        if w < self.min_size or h < self.min_size:
            return False, 0.0, f"Crop size too small ({w}x{h} < {self.min_size}px)"

        # Resize to normalized 128x128 for consistent Laplacian blur evaluation across distances
        resized = cv2.resize(crop_bgr, (128, 128), interpolation=cv2.INTER_AREA)
        # Convert to grayscale for Laplacian blur variance calculation
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if blur_score < self.min_blur_var:
            return False, blur_score, f"Blurry face crop (score {blur_score:.1f} < {self.min_blur_var:.1f})"

        return True, blur_score, "Passed"

quality_filter = FaceQualityFilter()
