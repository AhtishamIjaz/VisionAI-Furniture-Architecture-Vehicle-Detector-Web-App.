# VisionAI: Furniture, Architecture & Vehicle Detector (Offline Image Prediction)

A modern, fast, and intuitive offline computer vision web application designed specifically for **predicting objects from downloaded/uploaded photos** (No live webcams or video streams).

---

## 🚀 Key Features

1. **Pure Offline Image Prediction**:
   - Simply drag and drop or browse any picture downloaded from the internet or your camera roll.
   - Click **Run Prediction** to detect all objects.

2. **Specialized Detected Categories**:
   - **Furniture**: Chairs, Dining Tables, Beds, Couches/Sofas, Desks, Bookshelves, Wardrobes, Benches.
   - **Architecture & Interior**: Windows, Curtains, Glass/Doors, Houses, Buildings, Lamps, Mirrors.
   - **Vehicles**: Cars, Trucks, Buses, Motorcycles, Bicycles, Vans, Boats.

3. **Dual AI Model Engine**:
   - **YOLO-World (Open-Vocabulary)**: Detects fine-grained interior and architectural features like curtains, windows, glass, and houses.
   - **YOLOv8 Standard (COCO)**: High-accuracy detection for standard furniture, vehicles, and objects.

4. **Modern Interactive Dashboard**:
   - **Visual Comparison**: Toggle between Detection View (with color-coded boxes and confidence badges), Original Image, or Side-by-Side Split View.
   - **Statistics HUD**: Instant counters for Total Objects, Furniture, Architecture, Vehicles, and Inference Speed.
   - **Interactive Table**: Ranked object list with confidence % bars and coordinates.
   - **Export Tools**: Save annotated image (JPEG) or export the detection report (JSON).
   - **Bundled Sample Gallery**: Pre-loaded pictures for instant 1-click testing.

---

## 🏃 Quick Start

### Method 1: Double-Click Launcher (Easiest)
Double-click `run_app.bat` in this folder. It will start the server and automatically open your web browser at:
```
http://127.0.0.1:5000
```

### Method 2: Command Line
```bash
python app.py
```
Then open `http://127.0.0.1:5000` in Google Chrome or your favorite browser.

---

## 📂 Project Structure
```
VisionAI-Detector/
├── app.py                 # Flask web server & REST API
├── detector.py            # AI inference engine (YOLOv8 + YOLO-World)
├── run_app.bat            # 1-click batch launcher
├── models/                # Local weights (yolov8m.pt, yolov8s-worldv2.pt)
├── samples/               # Bundled sample downloaded images
├── static/
│   ├── css/style.css      # Dark glassmorphism UI styles
│   └── js/app.js          # Interactive frontend logic & AJAX
├── templates/
│   └── index.html         # Web application page
└── uploads/               # Uploaded images cache
```
