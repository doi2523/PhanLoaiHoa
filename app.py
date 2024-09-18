from flask import Flask, request, jsonify, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import shutil

app = Flask(__name__)

# Load mô hình phân loại ảnh
model = load_model('flowers_model.keras')

# Thư mục lưu ảnh tải lên
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Hàm chuẩn bị ảnh trước khi dự đoán
def prepare_image(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return 'Test route is working!'

@app.route('/predict', methods=['POST'])
def predict():
    # Xóa tất cả các tệp trong thư mục uploads trước khi xử lý tệp mới
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Xóa file
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Xóa thư mục (nếu có)
        except Exception as e:
            return jsonify({'error': f'Error deleting old files: {str(e)}'})

    # Kiểm tra file tải lên
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})
    
    files = request.files.getlist('file')

    if len(files) == 0:
        return jsonify({'error': 'No selected file'})
    if len(files) > 10:
        return jsonify({'error': 'Maximum 10 files allowed'})
    
    class_names = ['hoa cúc', 'bồ công anh', 'hoa hồng', 'hoa hướng dương', 'tulip']
    
    img_arrays = []
    results = []

    # Xử lý từng file ảnh
    for i, file in enumerate(files):
        img = Image.open(file.stream).convert('RGB')
        img_array = prepare_image(img)
        img_arrays.append(img_array)
        
        file_path = os.path.join(UPLOAD_FOLDER, f'image_{i}.jpg')
        img.save(file_path)
        
        img_info = {
            'filename': f'/uploads/image_{i}.jpg',
            'name': file.filename
        }
        results.append(img_info)

    img_arrays = np.vstack(img_arrays)
    
    try:
        # Dự đoán kết quả
        predictions = model.predict(img_arrays)
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'})

    # Gán dự đoán cho từng ảnh
    predicted_classes = [class_names[np.argmax(prediction)] for prediction in predictions]

    for img_info, pred_class in zip(results, predicted_classes):
        img_info['prediction'] = pred_class
    
    return jsonify({'results': results})

# Route để tải ảnh từ thư mục uploads
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
