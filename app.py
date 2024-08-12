from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(APP_ROOT, 'path_to_upload_folder') 

# Konfigurasi Flask-Session
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_session/'
Session(app)

app.config['UPLOAD_FOLDER'] = 'path_to_upload_folder'
app.config['ALLOWED_EXTENSIONS'] = {'xlsx'}

models = pickle.load(open('models.pkl', 'rb'))
rfc_model = models['RandomForest']
dt_model = models['DecisionTree']

def allowed_file(filename):
    allowed_extensions = app.config['ALLOWED_EXTENSIONS']
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def preprocess_data(file_path):
    data = pd.read_excel(file_path)
    feature = data.iloc[:, [data.columns.get_loc("Berat"), data.columns.get_loc("Tinggi"), data.columns.get_loc("Usia"), data.columns.get_loc("JK"), data.columns.get_loc("Status Gizi")]].copy()
    feature["JK"] = feature["JK"].astype("category")
    feature["JK"] = LabelEncoder().fit_transform(feature["JK"])
    feature["Status Gizi"] = feature["Status Gizi"].astype("category")
    feature["Status Gizi"] = LabelEncoder().fit_transform(feature["Status Gizi"])
    scaler = MinMaxScaler()
    scaler.fit(feature.drop(columns=['Status Gizi']))
    return data, feature, scaler

# Initialize with default data
data, feature, scaler = preprocess_data('data_tigaraksa.xlsx')

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/prediction")
def prediksi():
    return render_template('basic_elements.html')

@app.route("/data", methods=['GET', 'POST'])
def data_route():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            data, feature, scaler = preprocess_data(file_path)
            session['data'] = data.to_dict(orient='records')
            session['uploaded'] = True
            session['file_path'] = file_path  # Menyimpan path file di sesi
            session['filename'] = file.filename
            return redirect(url_for('data_route'))

    data_dict = session.get('data', [])
    uploaded = session.get('uploaded', False)

    return render_template('basic-table.html', data_dict=data_dict, uploaded=uploaded)

@app.route("/change")
def change_file():
    session.pop('data', None)
    session.pop('uploaded', None)
    session.pop('filename', None)
    return redirect(url_for('data_route'))

@app.route("/evaluasi")
def evaluasi():
    return render_template('chartjs.html')

# Set kamus untuk pemetaan hasil prediksi
keterangan_gizi = {
    0: 'Gizi Baik',
    1: 'Gizi Buruk',
    2: 'Gizi Kurang',
    3: 'Gizi Lebih',
    4: 'Obesitas',
    5: 'Risiko Gizi Lebih'
}

@app.route('/prediction', methods=["POST"])
def predict():
    data = request.form
    berat = float(data['berat'])
    tinggi = float(data['tinggi'])
    usia = int(data['usia'])
    jk = int(data['jk'])  # assuming 0 or 1

    # Create DataFrame from input data
    input_data = pd.DataFrame([[berat, tinggi, usia, jk]], columns=['Berat', 'Tinggi', 'Usia', 'JK'])

    input_data_scaled = scaler.transform(input_data)
    
    # Make predictions
    prediction_rf = rfc_model.predict(input_data_scaled)[0]
    prediction_tree = dt_model.predict(input_data_scaled)[0]

    # Map predictions to gizi descriptions
    keterangan_rfc = keterangan_gizi[prediction_rf]
    keterangan_tree = keterangan_gizi[prediction_tree]

    return render_template("basic_elements.html",
                           prediction_text_rfc="Prediksi Random Forest : {}".format(keterangan_rfc),
                           prediction_text_tree="Prediksi Decision Tree Classifier : {}".format(keterangan_tree))


if __name__ == "__main__":
    app.run(debug=True)
