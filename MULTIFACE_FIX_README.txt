Multi-face fixed version

What is fixed:
- Detects multiple faces using facenet-pytorch MTCNN
- Crops each face before age prediction
- Draws bounding boxes + age label on output image
- Shows average age and each face's age in result page
- Hides fake Unknown/Optional/0% fields for age-only model
- Adds face_details DB column automatically for existing SQLite DB

Run:
1) pip install -r requirements.txt
2) python app.py

Important:
Your model is age-only, so gender/emotion/confidence are not real and are hidden.
