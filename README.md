# 📦 AI-Based Smart Product Identification System
**Supervisor: Dr. Zaki Hassan | Group 13**
*Presented by Muhammad Aman Munir Abbasi & Syed Danish Javaid Gardazi*

---

## 📌 Project Overview
Traditional room classification models merely predict a coarse label (e.g. "Bedroom" or "Living Room") from an image without understanding **what is inside the room**. Supervisors and real-world applications require granular object-level intelligence.

This project delivers:
1. **Automated Furniture & Item Counting**: Detects and tallies individual objects (sofas, beds, chairs, dining tables, monitors, plants, appliances).
2. **Contextual Room Classification**: Synthesizes the detected object counts to determine the functional room category (Living Room, Bedroom, Dining Room, Kitchen, Office).
3. **Interactive Web Dashboard**: An upload-and-analyze interface with visual bounding boxes, category chips, and capacity metrics.
4. **Jupyter Notebook (`room_object_counter.ipynb`)**: Ready for academic demonstration, evaluation, and supervisor review.

---

## 🛠️ Architecture & Technologies
- **Deep Learning Model**: YOLOv11 / YOLOv8 ONNX optimized detector.
- **Inference Runtime**: `onnxruntime` & `OpenCV` (ultra-fast, CPU & GPU compatible, zero Torch DLL crashes).
- **Computer Vision**: Non-Maximum Suppression (NMS), aspect-ratio preserving letterboxing, dynamic bounding box rendering.
- **Backend**: Flask REST API.
- **Frontend**: Modern responsive dashboard with drag-and-drop file upload.

---

## 🚀 How to Run the Web Application

1. Open PowerShell or Command Prompt in this folder:
   ```powershell
   cd C:\Users\NexGen\Desktop\RoomAnalyzer
   ```

2. Start the server:
   ```powershell
   python app.py
   ```

3. Open your browser and navigate to:
   ```
   http://127.0.0.1:5050
   ```

4. Drag and drop any room photo:
   - The system immediately bounds and labels every detected object.
   - Shows the exact inventory: *"3 Sofas, 1 Chair, 4 Plants..."*
   - Predicts the room classification with confidence scores and reasoning.

---

## 📓 Running the Jupyter Notebook
If the supervisor asks to inspect code directly in Jupyter:
1. Launch Jupyter Notebook:
   ```powershell
   jupyter notebook room_object_counter.ipynb
   ```
2. Run all cells sequentially to see the step-by-step loading, inference, object counts, and matplotlib plot.

---

## 📊 Summary of Room Heuristics & Spatial Mapping
- **Bedroom**: Prioritizes `bed`, personal monitors/TVs.
- **Living Room**: Prioritizes multiple `couch`/sofas, coffee tables, lounge seating, TV.
- **Dining Room**: Prioritizes `dining table` surrounded by multi-count `chair`.
- **Kitchen**: Prioritizes appliances (`refrigerator`, `oven`, `microwave`, `sink`).
- **Office / Study**: Prioritizes `laptop`, desk surfaces, chairs, and study accessories.
