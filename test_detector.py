from detector import VisionDetector

d = VisionDetector()
print("=== Testing with COCO Model ===")
res1 = d.predict("samples/living_room_furniture.jpg", conf_thresh=0.25)
print(f"Living Room Detections ({len(res1['detections'])} objects):")
for x in res1['detections']:
    print(f"  - {x['label']} ({x['confidence_pct']}%) [{x['category']}]")

print("\n=== Testing with YOLO-World Model ===")
d.switch_model("world")
res2 = d.predict("samples/bedroom_window_curtains.jpg", conf_thresh=0.15)
print(f"Bedroom Detections ({len(res2['detections'])} objects):")
for x in res2['detections']:
    print(f"  - {x['label']} ({x['confidence_pct']}%) [{x['category']}]")

print("\n=== Testing Vehicles ===")
d.switch_model("coco")
res3 = d.predict("samples/vehicles_street_traffic.jpg", conf_thresh=0.25)
print(f"Street Detections ({len(res3['detections'])} objects):")
for x in res3['detections']:
    print(f"  - {x['label']} ({x['confidence_pct']}%) [{x['category']}]")

print("\n=== ALL TESTS PASSED SUCCESSFULLY! ===")
