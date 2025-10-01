from flask import Flask, render_template, request, redirect, url_for, session, flash
from datetime import datetime, timedelta
import os, random, cv2, torch
import numpy as np
#import pytesseract
import MySQLdb
import easyocr
import re

reader = easyocr.Reader(['en'])  # English OCR
# YOLOv5 imports
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.augmentations import letterbox

# === App Setup ===
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session login

# === Upload Paths ===
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
PLATE_FOLDER = 'static/plates'

for folder in [UPLOAD_FOLDER, RESULT_FOLDER, PLATE_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['PLATE_FOLDER'] = PLATE_FOLDER

# === Load YOLO Model ===
model = DetectMultiBackend('best.pt', device='cpu')
names = model.names

# === MySQL Connection ===
db = MySQLdb.connect(host="localhost", user="root", passwd="", db="anpr_db")
cursor = db.cursor()

# === ROUTES ===

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        staff_id = request.form['staff_id']
        password = request.form['password']

        cursor.execute("SELECT * FROM staffs WHERE staff_id=%s AND password=%s", (staff_id, password))
        staffs = cursor.fetchone()

        if staffs:
            session['staff_id'] = staff_id
            return redirect(url_for('home'))
        else:
            flash('Invalid ID or Password')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('staff_id', None)
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if 'staff_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if 'staff_id' not in session:
        return redirect(url_for('login'))

    uploaded_filename = None
    detected_filename = None
    plate_text = "N/A"
    vehicle_type = "N/A"
    crop_filename = ""

    if request.method == 'POST':
        file = request.files['image']
        if file:
            uploaded_filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + file.filename
            upload_path = os.path.join(UPLOAD_FOLDER, uploaded_filename)
            file.save(upload_path)

            img_clean = cv2.imread(upload_path)
            img_draw = img_clean.copy()

            img = letterbox(img_draw, new_shape=640)[0]
            img = img.transpose((2, 0, 1))[::-1]
            img = np.ascontiguousarray(img)
            img_tensor = torch.from_numpy(img).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)

            pred = model(img_tensor)
            pred = non_max_suppression(pred, 0.25, 0.45)

            for det in pred:
                if det is not None and len(det):
                    scaled_det = scale_coords(img_tensor.shape[2:], det[:, :4], img_draw.shape).round()
                    for i, det_item in enumerate(reversed(det)):
                        cls_name = names[int(det_item[5])].lower()
                        x1, y1, x2, y2 = map(int, scaled_det[len(det) - 1 - i])
                        label = f"{cls_name} {det_item[4]:.2f}"
                        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img_draw, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Plate detection
                        if cls_name == "number_plate" and plate_text == "N/A":
                           pad = 5
                           y1_pad = max(y1 - pad, 0)
                           y2_pad = min(y2 + pad, img_clean.shape[0])
                           x1_pad = max(x1 - pad, 0)
                           x2_pad = min(x2 + pad, img_clean.shape[1])
                           crop = img_clean[y1_pad:y2_pad, x1_pad:x2_pad]

                           if crop.size > 0:
                              # === Preprocessing + Upscaling ===
                              target_height = 100
                              scale_ratio = target_height / crop.shape[0]
                              resized = cv2.resize(crop, (int(crop.shape[1] * scale_ratio), target_height))

                              # Upscale for better OCR
                              resized = cv2.resize(resized, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                              gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                              # Bilateral filter preserves edges better than blur
                              gray = cv2.bilateralFilter(gray, 11, 17, 17)

                              # Adaptive threshold for varying lighting
                              thresh = cv2.adaptiveThreshold(
                                  gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2
                              )

                              # Morphological operations
                              kernel = np.ones((2, 2), np.uint8)
                              morph = cv2.dilate(thresh, kernel, iterations=1)
                              morph = cv2.erode(morph, kernel, iterations=1)
                                                
                              # OCR using EasyOCR (sorted left-to-right)
                              results = reader.readtext(resized, detail=1)

                              if results:
                                  if len(results) == 1:
                                       # Only one line detected (likely a car plate)
                                       plate_text = re.sub(r'[^A-Z0-9]', '', results[0][1]).upper()
                                  else:
                                       # More than one box detected – check if vertically aligned (bike plate) or horizontally aligned (car plate split)
                                       y_coords = [r[0][0][1] for r in results]  # top-left Y of each box
                                       y_range = max(y_coords) - min(y_coords)

                                       if y_range > 20:  # significantly different Y-coordinates → likely a bike (multi-line)
                                           sorted_results = sorted(results, key=lambda r: r[0][0][1])  # top to bottom
                                       else:
                                            sorted_results = sorted(results, key=lambda r: r[0][0][0])  # left to right

                                       combined_text = ''.join([r[1].replace(" ", "") for r in sorted_results])
                                       plate_text = re.sub(r'[^A-Z0-9]', '', combined_text).upper()
                              else:
                                  plate_text = "N/A"




                              plate_text = plate_text if plate_text else "N/A"

                              crop_filename = 'crop_' + uploaded_filename
                              cv2.imwrite(os.path.join(PLATE_FOLDER, crop_filename), crop)


                        # Vehicle detection
                        if cls_name in ["car", "bike"] and vehicle_type == "N/A":
                            vehicle_type = cls_name

            detected_filename = "det_" + uploaded_filename
            cv2.imwrite(os.path.join(RESULT_FOLDER, detected_filename), img_draw)

            entry_time = datetime.now()
            exit_time = entry_time + timedelta(minutes=random.randint(5, 180))
            cursor.execute("""
                INSERT INTO history (plate, plate_img, full_img, vehicle, entry_time, exit_time)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                plate_text,
                os.path.join('static/plates', crop_filename) if plate_text != "N/A" else "",
                upload_path,
                vehicle_type,
                entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                exit_time.strftime('%Y-%m-%d %H:%M:%S')
            ))
            db.commit()

    return render_template('recognize.html',
                           uploaded_filename=uploaded_filename,
                           detected_filename=detected_filename,
                           plate=plate_text,
                           vehicle=vehicle_type)

@app.route('/history')
def history():
    if 'staff_id' not in session:
        return redirect(url_for('login'))
    cursor.execute("SELECT * FROM history ORDER BY entry_time ASC")
    data = cursor.fetchall()
    return render_template('history.html', records=data)

@app.route('/delete/<int:record_id>', methods=['POST'])
def delete_record(record_id):
    if 'staff_id' not in session:
        return redirect(url_for('login'))
    
    # Delete the record
    cursor.execute("DELETE FROM history WHERE id = %s", (record_id,))
    db.commit()

    flash('Record deleted successfully.')
    return redirect(url_for('history'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
