import os
import json
import cloudinary
import cloudinary.uploader
from datetime import datetime
from functools import wraps
from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, send_file, jsonify)
from werkzeug.utils import secure_filename

from config import (
    SECRET_KEY, SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS,
    SQLALCHEMY_ENGINE_OPTIONS,
    UPLOAD_FOLDER, ALLOWED_EXTENSIONS, DEBUG,
    CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET, USE_CLOUDINARY
)

from models import db, User, Prediction, AdminLog
from predict_fixed import predict_image

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ── Configure Cloudinary ───────────────────────────────────
if USE_CLOUDINARY:
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
        secure=True,
    )

# ── Flask app ──────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"]                  = SECRET_KEY
app.config["SQLALCHEMY_DATABASE_URI"]     = SQLALCHEMY_DATABASE_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = SQLALCHEMY_TRACK_MODIFICATIONS
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = SQLALCHEMY_ENGINE_OPTIONS
app.config["UPLOAD_FOLDER"]              = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db.init_app(app)

# ── Helpers ────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def current_user():
    uid = session.get("user_id")
    return User.query.get(uid) if uid else None

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = current_user()
        if not user:
            flash("Please login first.", "error")
            return redirect(url_for("login"))
        if user.blocked:
            session.clear()
            flash("Your account has been blocked.", "error")
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = current_user()
        if not user or user.role != "admin":
            flash("Admin access required.", "error")
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

def create_default_admin():
    if not User.query.filter_by(email="admin@gmail.com").first():
        admin = User(name="Admin", email="admin@gmail.com", role="admin")
        admin.set_password("admin123")
        db.session.add(admin)
        db.session.commit()
        print("✅ Default admin created: admin@gmail.com / admin123")

@app.context_processor
def inject_globals():
    return {"auth_user": current_user(), "use_cloudinary": USE_CLOUDINARY}

# ── Public routes ──────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/features")
def features():
    return render_template("features.html")

@app.route("/demo")
def demo():
    return render_template("demo.html")

@app.route("/about")
def about():
    team = [
        {"name": "Md Raisul Islam",       "id": "231902037"},
        {"name": "Md Mahabub Hasan Mahin","id": "231902056"},
        {"name": "Chinmoy Debnath",       "id": "231902029"},
    ]
    return render_template("about.html", team=team)

# ── Auth ───────────────────────────────────────────────────
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        email    = request.form.get("email", "").lower().strip()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("All fields are required.", "error")
            return redirect(url_for("signup"))
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return redirect(url_for("signup"))
        if User.query.filter_by(email=email).first():
            flash("Email already registered. Please login.", "error")
            return redirect(url_for("signup"))

        user = User(name=name, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash("Account created! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form.get("email", "").lower().strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            flash("Invalid email or password.", "error")
            return redirect(url_for("login"))
        if user.blocked:
            flash("Your account is blocked. Contact admin.", "error")
            return redirect(url_for("login"))

        session["user_id"] = user.id
        flash(f"Welcome back, {user.name}! 👋", "success")
        return redirect(url_for("admin_dashboard") if user.role == "admin" else url_for("dashboard"))

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("home"))

# ── User routes ────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    user        = current_user()
    predictions = Prediction.query.filter_by(user_id=user.id)\
                                  .order_by(Prediction.created_at.desc()).all()
    return render_template("dashboard.html", predictions=predictions)

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("No image selected.", "error")
            return redirect(url_for("upload"))
        if not allowed_file(file.filename):
            flash("Only JPG, JPEG, PNG, WEBP allowed.", "error")
            return redirect(url_for("upload"))

        filename    = secure_filename(file.filename)
        unique_name = datetime.now().strftime("%Y%m%d%H%M%S_") + filename
        save_path   = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(save_path)

        annotated_name = "boxed_" + unique_name
        annotated_path = os.path.join(UPLOAD_FOLDER, annotated_name)

        # ── AI Prediction ──────────────────────────────────
        try:
            result = predict_image(save_path, annotated_output_path=annotated_path)
        except FileNotFoundError as e:
            flash(str(e), "error")
            return redirect(url_for("upload"))
        except Exception as e:
            flash(f"Prediction error: {e}", "error")
            return redirect(url_for("upload"))

        # Use annotated image if it was created
        display_name = annotated_name if os.path.exists(annotated_path) else unique_name
        display_path = annotated_path if os.path.exists(annotated_path) else save_path

        # ── Cloudinary upload ──────────────────────────────
        if USE_CLOUDINARY:
            try:
                cl_result  = cloudinary.uploader.upload(
                    display_path,
                    folder="age_ai_faceage",
                    public_id=display_name.rsplit(".", 1)[0],
                    overwrite=True,
                    resource_type="image",
                )
                image_url = cl_result["secure_url"]
                # Remove local copy to save space
                for p in [save_path, annotated_path]:
                    try: os.remove(p)
                    except: pass
            except Exception as e:
                flash(f"Cloudinary upload failed: {e}. Saving locally.", "error")
                image_url = url_for("static", filename=f"uploads/{display_name}")
        else:
            image_url = url_for("static", filename=f"uploads/{display_name}")

        # ── Save to DB ─────────────────────────────────────
        pred = Prediction(
            user_id      = current_user().id,
            image_url    = image_url,
            file_name    = display_name,
            predicted_age= result["predicted_age"],
            age_group    = result["age_group"],
            confidence   = result.get("confidence", 0.0),
            gender       = result.get("gender", ""),
            emotion      = result.get("emotion", ""),
            face_count   = result.get("face_count", 0),
            face_details = result.get("face_details", "[]"),
            message      = result.get("message", ""),
            mode         = result.get("mode", "opencv"),
        )
        db.session.add(pred)
        db.session.commit()
        return redirect(url_for("result", prediction_id=pred.id))

    return render_template("upload.html")

@app.route("/result/<int:prediction_id>")
@login_required
def result(prediction_id):
    pred = Prediction.query.get_or_404(prediction_id)
    user = current_user()
    if user.role != "admin" and pred.user_id != user.id:
        return redirect(url_for("dashboard"))

    try:
        face_details = json.loads(pred.face_details or "[]")
    except Exception:
        face_details = []

    return render_template("result.html", pred=pred, face_details=face_details)

@app.route("/delete-result/<int:prediction_id>", methods=["POST"])
@login_required
def delete_result(prediction_id):
    pred = Prediction.query.get_or_404(prediction_id)
    user = current_user()
    if user.role != "admin" and pred.user_id != user.id:
        return redirect(url_for("dashboard"))

    if USE_CLOUDINARY and "cloudinary.com" in pred.image_url:
        try:
            public_id = "age_ai_faceage/" + pred.file_name.rsplit(".", 1)[0]
            cloudinary.uploader.destroy(public_id)
        except Exception:
            pass
    else:
        try:
            path = os.path.join(BASE_DIR, "static", "uploads", pred.file_name)
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    db.session.delete(pred)
    db.session.commit()
    flash("Result deleted.", "success")
    return redirect(url_for("dashboard") if user.role != "admin" else url_for("admin_dashboard"))

@app.route("/download-result/<int:prediction_id>")
@login_required
def download_result(prediction_id):
    pred = Prediction.query.get_or_404(prediction_id)
    if "cloudinary.com" in pred.image_url:
        return redirect(pred.image_url)
    path = os.path.join(BASE_DIR, "static", "uploads", pred.file_name)
    if not os.path.exists(path):
        flash("File not found.", "error")
        return redirect(url_for("result", prediction_id=prediction_id))
    return send_file(path, as_attachment=True)

@app.route("/profile")
@login_required
def profile():
    user  = current_user()
    total = Prediction.query.filter_by(user_id=user.id).count()
    return render_template("profile.html", total_predictions=total)

# ── Admin routes ───────────────────────────────────────────
@app.route("/admin")
@admin_required
def admin_dashboard():
    users        = User.query.order_by(User.created_at.desc()).all()
    predictions  = Prediction.query.order_by(Prediction.created_at.desc()).all()
    logs         = AdminLog.query.order_by(AdminLog.created_at.desc()).limit(20).all()
    active_users = sum(1 for u in users if not u.blocked)
    blocked_users= sum(1 for u in users if u.blocked)

    storage = 0
    for root, dirs, files in os.walk(UPLOAD_FOLDER):
        for f in files:
            try: storage += os.path.getsize(os.path.join(root, f))
            except: pass

    return render_template(
        "admin.html",
        users=users, predictions=predictions, logs=logs,
        total_users=len(users), total_predictions=len(predictions),
        storage_mb=round(storage / (1024 * 1024), 2),
        active_users=active_users, blocked_users=blocked_users,
        recent_uploads=predictions[:8], use_cloudinary=USE_CLOUDINARY,
    )

@app.route("/admin/block/<int:user_id>", methods=["POST"])
@admin_required
def block_user(user_id):
    user = User.query.get_or_404(user_id)
    user.blocked = not user.blocked
    log = AdminLog(
        admin_id=current_user().id,
        action=f"{'Blocked' if user.blocked else 'Unblocked'} user: {user.email}",
        target_user_id=user.id
    )
    db.session.add(log)
    db.session.commit()
    flash(f"User {'blocked' if user.blocked else 'unblocked'}.", "success")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/delete-user/<int:user_id>", methods=["POST"])
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.role == "admin":
        flash("Cannot delete admin account.", "error")
        return redirect(url_for("admin_dashboard"))
    log = AdminLog(
        admin_id=current_user().id,
        action=f"Deleted user: {user.email}",
        target_user_id=user.id
    )
    db.session.add(log)
    db.session.delete(user)
    db.session.commit()
    flash("User deleted.", "success")
    return redirect(url_for("admin_dashboard"))

# ── DB schema migration (SQLite safe) ─────────────────────
def ensure_schema():
    try:
        if "sqlite" not in SQLALCHEMY_DATABASE_URI:
            return  # skip for Supabase/Postgres

        with db.engine.connect() as conn:
            rows = conn.exec_driver_sql("PRAGMA table_info(predictions)").fetchall()
            cols = {row[1] for row in rows}

            new_cols = {
                "face_details": "TEXT",
                "report_url": "VARCHAR(300)",
                "message": "VARCHAR(300)",
                "mode": "VARCHAR(40)",
            }

            for col, col_type in new_cols.items():
                if col not in cols:
                    conn.exec_driver_sql(f"ALTER TABLE predictions ADD COLUMN {col} {col_type}")

            conn.commit()
    except Exception:
        pass

# ── App startup ────────────────────────────────────────────
with app.app_context():
    db.create_all()
    ensure_schema()
    create_default_admin()

@app.route("/index.html")
@app.route("/index")
def index_launcher():
    import os as _os
    p = _os.path.join(BASE_DIR, "index.html")
    if _os.path.exists(p):
        return send_file(p)
    return redirect(url_for("home"))

if __name__ == "__main__":
    print(f"🚀 Starting AI FaceAge | Debug={DEBUG} | Cloudinary={USE_CLOUDINARY}")

    if DEBUG:
        print(f"   DB: {SQLALCHEMY_DATABASE_URI[:40]}...")

    app.run(debug=DEBUG, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
