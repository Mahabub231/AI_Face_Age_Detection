import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Flask ────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-in-production")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# ── AI Mode ─────────────────────────────────────────────
AGE_AI_MODE = os.getenv("AGE_AI_MODE", "opencv")

# ── Database ────────────────────────────────────────────
db_url = os.getenv("DATABASE_URL", "").strip()

if db_url:
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    if "sslmode=" not in db_url:
        joiner = "&" if "?" in db_url else "?"
        db_url = f"{db_url}{joiner}sslmode=require"

    SQLALCHEMY_DATABASE_URI = db_url
else:
    # fallback (important for safety)
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(BASE_DIR, "database.db")

# REQUIRED (missing before)
SQLALCHEMY_TRACK_MODIFICATIONS = False

SQLALCHEMY_ENGINE_OPTIONS = {
    "pool_pre_ping": True,
    "pool_recycle": 280,
}

# ── Upload ──────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# ── Supabase ────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# ── Cloudinary ──────────────────────────────────────────
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "").strip()
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "").strip()
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "").strip()

USE_CLOUDINARY = bool(
    CLOUDINARY_CLOUD_NAME
    and CLOUDINARY_API_KEY
    and CLOUDINARY_API_SECRET
    and " " not in CLOUDINARY_CLOUD_NAME
)