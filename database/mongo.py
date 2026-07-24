import logging
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Optional, Mapping
from motor.motor_asyncio import AsyncIOMotorClient
from config import settings

logger = logging.getLogger("mongo_db")

def _sanitize_bson(obj: object) -> object:
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return obj.item()
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): _sanitize_bson(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_bson(x) for x in obj]
    return obj

class MongoDBManager:
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None

    async def connect(self):
        try:
            self.client = AsyncIOMotorClient(settings.MONGO_URI, serverSelectionTimeoutMS=3000)
            self.db = self.client[settings.MONGO_DB_NAME]
            # Ping database
            await self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB at {settings.MONGO_URI}")
            await self._setup_indexes()
        except Exception as e:
            logger.warning(f"MongoDB connection warning: {e}. Running in memory fallback mode.")

    async def _setup_indexes(self):
        if self.db is None:
            return
        try:
            # TTL Index on detection_events timestamp for automatic retention cleanup
            events_coll = self.db["detection_events"]
            await events_coll.create_index(
                "timestamp",
                expireAfterSeconds=settings.RETENTION_DAYS * 86400,
                background=True
            )
            # Index on profile_id and camera_id for fast queries
            await events_coll.create_index("profile_id", background=True)
            await events_coll.create_index("camera_id", background=True)
            
            profiles_coll = self.db["face_profiles"]
            await profiles_coll.create_index("profile_id", unique=True, background=True)
            logger.info("MongoDB indexes verified successfully.")
        except Exception as e:
            logger.error(f"Error setting up MongoDB indexes: {e}")

    async def close(self):
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed.")

    async def save_detection_event(self, event_data: Mapping[str, object]) -> bool:
        if self.db is None:
            return False
        try:
            clean_data = _sanitize_bson(event_data)
            if isinstance(clean_data, dict):
                await self.db["detection_events"].insert_one(clean_data)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to insert detection event: {e}")
            return False

    async def get_recent_events(self, limit: int = 50, camera_id: Optional[str] = None) -> List[Dict[str, object]]:
        if self.db is None:
            return []
        try:
            query = {}
            if camera_id:
                query["camera_id"] = camera_id
            cursor = self.db["detection_events"].find(query).sort("timestamp", -1).limit(limit)
            events = []
            async for doc in cursor:
                doc["_id"] = str(doc["_id"])
                events.append(doc)
            return events
        except Exception as e:
            logger.error(f"Failed to fetch detection events: {e}")
            return []

    async def save_or_update_profile(self, profile_id: str, embedding: List[float], name: Optional[str] = None) -> bool:
        if self.db is None:
            return False
        try:
            profiles_coll = self.db["face_profiles"]
            existing = await profiles_coll.find_one({"profile_id": profile_id})
            now = datetime.now(timezone.utc)
            
            if existing:
                embeddings = existing.get("embeddings", [])
                if len(embeddings) < settings.MAX_GALLERY_EMBEDDINGS:
                    embeddings.append(embedding)
                    await profiles_coll.update_one(
                        {"profile_id": profile_id},
                        {
                            "$set": {
                                "embeddings": embeddings,
                                "last_seen": now,
                                "sample_count": len(embeddings)
                            }
                        }
                    )
                else:
                    await profiles_coll.update_one(
                        {"profile_id": profile_id},
                        {"$set": {"last_seen": now}}
                    )
            else:
                doc = {
                    "profile_id": profile_id,
                    "name": name or f"Identity {profile_id}",
                    "created_at": now,
                    "last_seen": now,
                    "embeddings": [embedding],
                    "sample_count": 1,
                    "status": "active"
                }
                await profiles_coll.insert_one(doc)
            return True
        except Exception as e:
            logger.error(f"Failed to save profile {profile_id}: {e}")
            return False

    async def load_all_profiles(self) -> List[Dict[str, object]]:
        if self.db is None:
            return []
        try:
            cursor = self.db["face_profiles"].find({"status": "active"})
            profiles = []
            async for doc in cursor:
                doc["_id"] = str(doc["_id"])
                profiles.append(doc)
            return profiles
        except Exception as e:
            logger.error(f"Failed to load profiles: {e}")
            return []

mongo_db = MongoDBManager()
