import os
import cv2
import logging
import shutil
from datetime import datetime, timedelta
from config import settings

logger = logging.getLogger("crop_storage")

class CropStorageManager:
    def __init__(self, base_dir: str = settings.CROP_DIR):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save_crop(self, crop_bgr, profile_id: str, track_id: int) -> str:
        """
        Saves a cropped BGR face image to disk inside date and profile directory.
        Returns the relative file path.
        """
        try:
            today_str = datetime.now().strftime("%Y-%m-%d")
            folder_path = os.path.join(self.base_dir, today_str, profile_id)
            os.makedirs(folder_path, exist_ok=True)

            filename = f"track_{track_id}_{int(datetime.now().timestamp())}.jpg"
            file_path = os.path.join(folder_path, filename)

            # Compress slightly to save disk space
            cv2.imwrite(file_path, crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return file_path
        except Exception as e:
            logger.error(f"Error saving face crop for profile {profile_id}: {e}")
            return ""

    def prune_old_crops(self, retention_days: int = settings.RETENTION_DAYS):
        """
        Deletes date directories in storage older than retention_days.
        """
        try:
            now = datetime.now()
            cutoff_date = now - timedelta(days=retention_days)

            for item in os.listdir(self.base_dir):
                item_path = os.path.join(self.base_dir, item)
                if os.path.isdir(item_path):
                    try:
                        folder_date = datetime.strptime(item, "%Y-%m-%d")
                        if folder_date < cutoff_date:
                            shutil.rmtree(item_path)
                            logger.info(f"Pruned expired crop directory: {item_path}")
                    except ValueError:
                        # Skip directories not following YYYY-MM-DD pattern
                        continue
        except Exception as e:
            logger.error(f"Error pruning crop storage: {e}")

crop_storage = CropStorageManager()
