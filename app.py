from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import numpy as np
import scipy.io as sio
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Konfigurasi folder
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static'
MODEL_DIR = 'model'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MODEL_DIR'] = MODEL_DIR

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load semua model TensorFlow saat aplikasi dimulai
models = {}
for model_file in os.listdir(MODEL_DIR):
    model_path = os.path.join(MODEL_DIR, model_file)
    models[model_file[:-3]] = load_model(model_path, compile=False)

# Validasi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'mat'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "File tidak ditemukan dalam permintaan", 400
        
        file = request.files['file']
        if file.filename == '':
            return "Tidak ada file yang dipilih", 400
        
        if not allowed_file(file.filename):
            return "Hanya file .mat yang diperbolehkan", 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load data dari file .mat
        try:
            dmat = sio.loadmat(filepath)
        except Exception as e:
            return f"Error saat memproses file: {str(e)}", 400

        # Ambil array numpy dari file .mat
        arr = None
        for key in dmat.keys():
            if isinstance(dmat[key], np.ndarray):
                arr = dmat[key]
                break
        
        if arr is None:
            return "File tidak memiliki array numpy yang valid.", 400

        # Downsampling data
        arr_small = arr[::4, ::4, ::4]

        # Membuat grid 3D untuk visualisasi
        X, Y, Z = np.mgrid[0:arr_small.shape[0], 0:arr_small.shape[1], 0:arr_small.shape[2]]

        # Membuat visualisasi Plotly
        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=arr_small.flatten(),
            isomin=0.5,
            isomax=1.0,
            surface_count=10,
            colorscale='RdBu'
        ))

        # Konversi plot ke HTML
        plot_html = pio.to_html(fig, full_html=False)

        # Pisahkan Channel RGB
        rgb = np.zeros([128, 128, 3], dtype=np.uint8) 

        arxy = arr[:, :, 64]
        aryz = arr[64, :, :]
        arzx = arr[:, 64, :]

        # Scale to 0-255 for image format
        xy = (arxy * 255).astype(np.uint8)
        yz = (aryz * 255).astype(np.uint8)
        zx = (arzx * 255).astype(np.uint8)

        # Assemble RGB array
        rgb[:, :, 0] = xy
        rgb[:, :, 1] = yz
        rgb[:, :, 2] = zx

        # Save RGB image
        img_rgb = Image.fromarray(rgb)
        rgb_path = os.path.join(app.config['RESULT_FOLDER'], 'result_rgb.png')
        img_rgb.save(rgb_path)

        # Save channel images
        img_xy = Image.fromarray(xy)
        img_yz = Image.fromarray(yz)
        img_zx = Image.fromarray(zx)

        xy_path = os.path.join(app.config['RESULT_FOLDER'], 'result_xy.png')
        yz_path = os.path.join(app.config['RESULT_FOLDER'], 'result_yz.png')
        zx_path = os.path.join(app.config['RESULT_FOLDER'], 'result_zx.png')

        img_xy.save(xy_path)
        img_yz.save(yz_path)
        img_zx.save(zx_path)

        # Mengirimkan plot dan informasi array ke template
        return render_template('index.html',
                               plot_html=plot_html,
                               rgb_path=rgb_path,
                               xy_path=xy_path,
                               yz_path=yz_path,
                               zx_path=zx_path,
                               file=file)

    # Jika GET request, tampilkan form upload
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def array_info():
    # Ambil nama file dari frontend
    file_name = request.json.get('file_path')  
    if not file_name:
        return jsonify({
            "status": "error",
            "message": "File name not provided."
        }), 400

    # Gabungkan nama file dengan direktori 'uploads'
    file_path = os.path.join(app.config['RESULT_FOLDER'], 'result_rgb.png')
    print(file_path)
    if not os.path.exists(file_path):
        return jsonify({
            "status": "error",
            "message": f"File not found at path: {file_path}"
        }), 400
    
    # Baca gambar RGB
    try:
        img_rgb = Image.open(file_path)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error saat membaca file: {str(e)}"
        }), 400

    # Preprocessing untuk model prediksi
    ars = np.reshape(np.array(img_rgb), (1, 128, 128, 3))
    sampel = ars / 255

    # Lakukan prediksi dengan model TensorFlow
    model_dir = 'model/'
    folder = os.listdir(model_dir)
    results = {}
    coef = [1, 0.444, 0.444, 0.01, 444.44]
    units = ['', 'px', 'px', '', 'px\u207B\u00B9']

    for idx, model_file in enumerate(folder):
        model_path = os.path.join(model_dir, model_file)
        model = load_model(model_path, compile=False)
        prediksi = model(sampel)
        print(f"{model_file[:-3]}: finished")
        
        # Mengonversi prediksi menjadi angka float
        formatted_prediksi = float(prediksi.numpy().astype(float).flatten()[0])
        
        # Menambahkan hasil prediksi ke dictionary dengan format yang diinginkan
        results[model_file[:-3]] = f"{formatted_prediksi * coef[idx]:.2f} {units[idx]}"
    
    # Return hasil prediksi
    return jsonify({
        "status": "success",
        "message": "Prediction completed",
        "predictions": results
    }), 200



if __name__ == '__main__':
    app.run(debug=True)
