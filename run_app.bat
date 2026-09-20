@echo off
title VisionAI - Furniture, Architecture & Vehicle Detector
echo ================================================================
echo      VisionAI: Offline Picture Object Detection System
echo      Detects Furniture (Chairs, Tables, Beds), 
echo      Architecture (Windows, Curtains, Houses), Vehicles (Cars, Trucks)
echo ================================================================
echo.
echo Starting web server on http://127.0.0.1:5000 ...
start "" http://127.0.0.1:5000
python app.py
pause
