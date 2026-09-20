from app import app
import io

client = app.test_client()

# Test 1: GET /
res_index = client.get('/')
assert res_index.status_code == 200, f'Index failed: {res_index.status_code}'
print('[PASS] GET / -> 200 OK')

# Test 2: GET /api/samples
res_samples = client.get('/api/samples')
assert res_samples.status_code == 200, f'Samples failed: {res_samples.status_code}'
samples_json = res_samples.get_json()
print(f"[PASS] GET /api/samples -> {len(samples_json['samples'])} samples found")

# Test 3: POST /api/predict with sample
res_pred = client.post('/api/predict', data={'sample': 'living_room_furniture.jpg', 'conf': '0.25', 'model': 'coco'})
assert res_pred.status_code == 200, f'Predict failed: {res_pred.status_code}'
pred_json = res_pred.get_json()
assert pred_json['success'] is True
print(f"[PASS] POST /api/predict (Sample) -> Found {pred_json['total_objects']} objects in {pred_json['inference_time_ms']}ms")

# Test 4: POST /api/predict with uploaded file
with open('samples/bedroom_window_curtains.jpg', 'rb') as f:
    img_bytes = f.read()

res_upload = client.post('/api/predict', data={
    'file': (io.BytesIO(img_bytes), 'bedroom.jpg'),
    'conf': '0.15',
    'model': 'world'
}, content_type='multipart/form-data')
assert res_upload.status_code == 200, f'Upload failed: {res_upload.status_code}'
upload_json = res_upload.get_json()
assert upload_json['success'] is True
print(f"[PASS] POST /api/predict (File Upload) -> Found {upload_json['total_objects']} objects in {upload_json['inference_time_ms']}ms")
for obj in upload_json['detections']:
    print(f"   -> {obj['label']} ({obj['confidence_pct']}%) [{obj['category']}]")

print('\nALL END-TO-END BACKEND & API TESTS PASSED 100%!')
