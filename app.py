from flask import Flask, request, jsonify, render_template
import os
import calendar
import datetime
# import cv2
import pickle
import urllib.request
import json
import pyrebase
import requests
import math
from sklearn import neighbors
import os.path
from PIL import Image, ImageDraw, ImageFont
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import time
import numpy as np
import collections
from secrets import config_fir, pwd

# Init app
app = Flask(__name__)

# Init Firebase
config = {
    "apiKey": config_fir['apiKey'],
    "authDomain": config_fir['authDomain'],
    "databaseURL": config_fir['databaseURL'],
    "projectId": config_fir['projectId'],
    "storageBucket": config_fir['storageBucket'],
    "messagingSenderId": config_fir['messagingSenderId'],
    "appId": config_fir['appId']
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
database = firebase.database()

# Init directory
BASEDIR = os.path.abspath(os.path.dirname(__file__))

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# # fungsi rescale image cv2
# def scale_down(scale, img):
#     width = int(img.shape[1] * scale/100)
#     height = int(img.shape[0] * scale/100)
#     dim = (width, height)
#     new_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

#     return new_image

# fungsi cek_size
def cek_size(size, img, path_name, file_name):
    if size > 100000 and size < 200000:
        scale_percent = 70
        # rescale = scale_down(scale_percent, img)
        path_save = os.path.join(path_name, file_name)
        # cv2.imwrite(path_save, rescale)
        resizedImg = img.resize((round(img.size[0]*(scale_percent/100)), round(img.size[1]*(scale_percent/100)))) # resize use PIL Image
        resizedImg.save(path_save)
        new_size = os.path.getsize(path_save)
        print(f"New Size: {new_size/1024} (100)")
    elif size > 200000 and size < 300000:
        scale_percent = 60
        # rescale = scale_down(scale_percent, img)
        path_save = os.path.join(path_name, file_name)
        # cv2.imwrite(path_save, rescale)
        resizedImg = img.resize((round(img.size[0]*(scale_percent/100)), round(img.size[1]*(scale_percent/100)))) # resize use PIL Image
        resizedImg.save(path_save)
        new_size = os.path.getsize(path_save)
        print(f"New Size: {new_size/1024} (200)")
    elif size > 300000 and size < 400000:
        scale_percent = 50
        # rescale = scale_down(scale_percent, img)
        path_save = os.path.join(path_name, file_name)
        # cv2.imwrite(path_save, rescale)
        resizedImg = img.resize((round(img.size[0]*(scale_percent/100)), round(img.size[1]*(scale_percent/100)))) # resize use PIL Image
        resizedImg.save(path_save)
        new_size = os.path.getsize(path_save)
        print(f"New Size: {new_size/1024} (300)")
    elif size > 400000 and size < 500000:
        scale_percent = 40
        # rescale = scale_down(scale_percent, img)
        path_save = os.path.join(path_name, file_name)
        # cv2.imwrite(path_save, rescale)
        resizedImg = img.resize((round(img.size[0]*(scale_percent/100)), round(img.size[1]*(scale_percent/100)))) # resize use PIL Image
        resizedImg.save(path_save)
        new_size = os.path.getsize(path_save)
        print(f"New Size: {new_size/1024} (400)")
    elif size > 500000:
        scale_percent = 30
        # rescale = scale_down(scale_percent, img)
        path_save = os.path.join(path_name, file_name)
        # cv2.imwrite(path_save, rescale)
        resizedImg = img.resize((round(img.size[0]*(scale_percent/100)), round(img.size[1]*(scale_percent/100)))) # resize use PIL Image
        resizedImg.save(path_save)
        new_size = os.path.getsize(path_save)
        print(f"New Size: {new_size/1024} (500++)")

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):

    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            # resize image
            # img = cv2.imread(img_path) # img from cv2 (opencv)
            img = Image.open(img_path)
            size = os.path.getsize(img_path)
            file_name = os.path.basename(img_path)
            path_name = os.path.dirname(img_path)
            cek_size(size, img, path_name, file_name)
            # load image use face_recognition
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image) # default model="hog", pilihan lain "cnn"

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir) # nama
                print(f"[TRAIN] {path_name} - {file_name}")
    
    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)
    
    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):

    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    
    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    # add number of times to upsample utk mendeteksi wajah yang jauh dari kamera
    # X_face_locations = face_recognition.face_locations(X_img, number_of_times_to_upsample=2) # default model="hog", pilihan lain "cnn"
    X_face_locations = face_recognition.face_locations(X_img) # testing utk single face

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []
    
    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    distances = [closest_distances[0][i][0] for i in range(len(X_face_locations))] # distance yg didapat

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc, distance) if rec else ("unknown", loc, distance) for pred, loc, rec, distance in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches, distances)]

def show_prediction_labels_on_image(img_path, predictions, kelas, path_hasil_uji_presensi, presensi):

    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    # Simpan crop gambar utk validasi di Android (Upload ke firebase, get url image)
    load_image = face_recognition.load_image_file(img_path)

    for name, (top, right, bottom, left), dist in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0))
        # face_image = load_image[top:bottom, left:right]
        # crop_image = Image.fromarray(face_image)
        # # simpan crop
        # print(f"[CROP] {name}")
        # # save di hasil uji
        # crop_image.save(f"{path_hasil_uji_presensi}/{name}.jpg")
        # # save di firebase kelas/hasil/id_presensi
        # # download url menggunakan NAMA MAHASISWA
        # storage.child("uploads/kelas/hasil/"+presensi+"/"+name+".jpg").put(path_hasil_latih+"/"+name+".jpg")
        # data = {
        #     name : name+".jpg"
        # }
        # database.child("uploads/kelas/hasil/"+presensi).child("hasil").set(data)

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        # name = name.encode("UTF-8")

        font = ImageFont.truetype(font="OpenSansSemibold.ttf", size=11)
        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(255, 255, 0), outline=(225, 225, 0))
        draw.text((left + 6, bottom - text_height - 7), name, fill=(0, 0, 0), font=font)
    
    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    # pil_image.show()

    # Save Image di Local dan Firebase
    unix = time.time()
    if kelas is not None: # untuk hasil foto kelas / multiple face
        if not os.path.exists(path_hasil_uji_presensi+"/foto"):
            os.mkdir(path_hasil_uji_presensi+"/foto")
        pil_image.save(path_hasil_uji_presensi+"/foto/kelas_"+kelas+"_"+str(unix)+".jpg")
        # save firebase
        upload = storage.child("uploads/kelas/hasil/"+presensi+"/foto/kelas_"+kelas+"_"+str(unix)+".jpg").put(path_hasil_uji_presensi+"/foto/kelas_"+kelas+"_"+str(unix)+".jpg")
        # get url gambar, simpan di url_gambar.txt
        get_url = storage.child("uploads/kelas/hasil/"+presensi+"/foto/kelas_"+kelas+"_"+str(unix)+".jpg").get_url(upload['downloadTokens'])
        data = {
            "kelas" : "kelas_"+kelas+"_"+str(unix)+".jpg",
            "url" : get_url
        }
        database.child("uploads/kelas/hasil/"+presensi).child("foto").push(data)
        
    else: # untuk hasil single auth
        pil_image.save(path_hasil_uji_presensi+"/"+presensi+"_"+str(unix)+".jpg")
        # save firebase

@app.route('/')
def index():
    return render_template('index.html', awal='index')

@app.route('/listkelas', methods=['POST'])
def listkelas():
    if request.method == 'POST':
        sandi = request.form['sandi']
        if sandi == pwd:
            kelas = []
            matkul = []
            
            # request get data mengajar
            req_mengajar_l = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-mengajar')
            json_req_mengajar_l = req_mengajar_l.json()
            master_req_mengajar_l = json_req_mengajar_l['master']
            # return jsonify(master_req_mengajar_l)
            
            # list comprehension
            [kelas.append(i['kelas']) for i in master_req_mengajar_l if i['kelas'] not in kelas]
            [matkul.append(j['matakuliah']) for j in master_req_mengajar_l if j['matakuliah'] not in matkul]

            kelas.sort()
            matkul.sort()

            return render_template('index.html', kelas=kelas, matkul=matkul)
        else:
            return render_template('index.html', message='Error - Kamu siapa ?')
    else:
        return render_template('index.html', message='Error - Isi Form dulu')

# Latih Semua Data
@app.route('/latihsemua', methods=['GET'])
def latihsemua():
    if request.method == 'GET':
        dir_single2 = os.path.join(BASEDIR, "single_v2") # path single face
        if not os.path.exists(dir_single2):
            os.mkdir(dir_single2)
        
        # path_spresensi = os.path.join(dir_single2, str(presensi))
        # if not os.path.exists(path_spresensi):
        #     os.mkdir(path_spresensi)
        #     print("- Create",path_spresensi)
        
        path_data = os.path.join(dir_single2, "data") # single2/data
        if not os.path.exists(path_data):
            os.mkdir(path_data)

        path_hasil = os.path.join(dir_single2, "hasil") # single2/hasil
        if not os.path.exists(path_hasil):
            os.mkdir(path_hasil)
        
        path_data_latih = os.path.join(path_data, "latih") # single2/data/latih
        if not os.path.exists(path_data_latih):
            os.mkdir(path_data_latih)
        
        path_data_uji = os.path.join(path_data, "uji") # single2/data/uji
        if not os.path.exists(path_data_uji):
            os.mkdir(path_data_uji)
        
        path_hasil_latih = os.path.join(path_hasil, "latih") # single2/hasil/latih
        if not os.path.exists(path_hasil_latih):
            os.mkdir(path_hasil_latih)
        
        path_hasil_uji = os.path.join(path_hasil, "uji") # single2/hasil/uji
        if not os.path.exists(path_hasil_uji):
            os.mkdir(path_hasil_uji)
        
        # path_data_uji_presensi = os.path.join(path_data_uji, str(presensi)) # single2/data/uji/presensi
        # if not os.path.exists(path_data_uji_presensi):
        #     os.mkdir(path_data_uji_presensi)
        
        # path_hasil_uji_presensi = os.path.join(path_hasil_uji, str(presensi)) # single2/hasil/uji/presensi
        # if not os.path.exists(path_hasil_uji_presensi):
        #     os.mkdir(path_hasil_uji_presensi)

        # path_data_uji_p_key = os.path.join(path_data_uji_presensi, str(key)) # single2/data/uji/presensi/key
        # if not os.path.exists(path_data_uji_p_key):
        #     os.mkdir(path_data_uji_p_key)
        # else: # hapus data lama, utk prediksi satu foto uji saja
        #     for files in os.listdir(path_data_uji_p_key):
        #         full_path = os.path.join(path_data_uji_p_key, files)
        #         print(f"[INFO] Hapus {full_path}")
        #         os.remove(full_path)
        
        # path_hasil_uji_p_key = os.path.join(path_hasil_uji_presensi, str(key)) # single2/hasil/uji/presensi/key
        # if not os.path.exists(path_hasil_uji_p_key):
        #     os.mkdir(path_hasil_uji_p_key)

        # latih dan save model 

        if not os.path.exists(path_hasil_latih+"/trained_knn_model.clf"):
            # data = []
            # data.append({
            #     "nama" : nama,
            #     "nim" : nim,
            #     "url" : get_url,
            #     "url_nama" : get_url_nama
            # })
            # return jsonify(data)

            # download data latih dan train beda route
            # get nim from firebase
            data_mhs = database.child("uploads").get()
            for x in data_mhs.each():
                nim_f = x.key()
                if len(nim_f) == 11:
                    req_mahasiswa = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-mahasiswa?mahasiswa='+str(nim_f))
                    json_req_mahasiswa = req_mahasiswa.json()
                    master_req_mahasiswa = json_req_mahasiswa['master']
                    nama_mhs = master_req_mahasiswa[0]['nama'] # nama berdasarkan nim

                    path_data_latih_nama = os.path.join(path_data_latih, nama_mhs) # single/data/latih/nama
                    if not os.path.exists(path_data_latih_nama):
                        os.mkdir(path_data_latih_nama)

                    # download foto dari firebase
                    for i in x.val():
                        url = x.val()[i]['imageUrl']
                        file_name = x.val()[i]['name']
                        if not os.path.exists(path_data_latih_nama+"/"+i+"-"+file_name+".jpg"):
                            urllib.request.urlretrieve(url, path_data_latih_nama+"/"+i+"-"+file_name+".jpg") # nama file yg disimpan
                            # download gambar yg telah diupload mahasiswa
                            print("[SUCCESS] download "+file_name+".jpg berhasil")
                        else:
                            print(f"[TERSEDIA] {file_name}.jpg ")

            klasifikasi = train(path_data_latih, model_save_path=path_hasil_latih+"/trained_knn_model.clf", n_neighbors=2)
            print("Data Training Selesai dibuat")
            return jsonify("Data Training Selesai")
        else:
            print("Data Training Sudah Ada")
        return render_template('index.html', message='Data Latih Selesai')

@app.route('/buatlatih', methods=['POST'])
def buatlatih():
    if request.method == 'POST':
        start_time = time.time()
        kelas = request.form['kelas']
        matkul = request.form['matkul']

        if kelas == '' or matkul == '':
            return render_template(
                'index.html',
                message='Data form kosong'
            )

        # select id kelas
        req_kelas = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-kelas?kelas='+str(kelas))
        json_req_kelas = req_kelas.json()
        master_req_kelas = json_req_kelas['master']
        if not len(master_req_kelas) == 0: 
            id_kelas = master_req_kelas[0]['id_kelas']
        else:
            return render_template(
                'index.html',
                message='Error - id kelas'
            )

        # select id matkul
        req_matkul = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-matkul?matkul='+str(matkul))
        json_req_matkul = req_matkul.json()
        master_req_matkul = json_req_matkul['master']
        if not len(master_req_matkul) == 0:
            id_matkul = master_req_matkul[0]['id_matakuliah']
        else:
            return render_template(
                'index.html',
                message='Error - id mata kuliah'
            )

        # select id mengajar
        # revisi, select id mengajar, juga harus pakai id_mata_kuliah, get semester aktif
        req_mengajar = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-dosen?dosen='+str(id_kelas)+'&matkul='+str(id_matkul))
        json_req_mengajar = req_mengajar.json()
        master_req_mengajar = json_req_mengajar['master']
        if not len(master_req_mengajar) == 0:
            id_mengajar = master_req_mengajar[0]['id_mengajar']
        else:
            print(json_req_mengajar)
            return render_template(
                'index.html',
                message='Error - Kelas dan matkul tidak tersedia'
            )

        # select nim mahasiswa 
        req_mengambil = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-mhs?mhs='+str(id_mengajar))
        json_req_mengambil = req_mengambil.json()
        master_req_mengambil = json_req_mengambil['master']
        # return jsonify(id_mengajar)

        # insert data presensi baru
        # select id_presensi & pertemuan from tb_presensi
        # request ini tidak dipakai dulu
        req_pertemuan = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-presensi?presensi='+str(id_mengajar))
        json_req_pertemuan = req_pertemuan.json()
        master_req_pertemuan = json_req_pertemuan['master']

        dir_kelas = "kelas_"+kelas+"_"+matkul
        path_kelas = os.path.join(BASEDIR, dir_kelas)

        if not os.path.exists(path_kelas):
            print(f"[CREATED] {path_kelas}")
            os.mkdir(path_kelas)
        
        path_data = os.path.join(path_kelas, "data")
        if not os.path.exists(path_data):
            os.mkdir(path_data)

        path_data_latih = os.path.join(path_data, "latih")
        if not os.path.exists(path_data_latih):
            os.mkdir(path_data_latih)

        path_hasil = os.path.join(path_kelas, "hasil")
        if not os.path.exists(path_hasil):
            os.mkdir(path_hasil)

        path_hasil_latih = os.path.join(path_hasil, "latih")
        if not os.path.exists(path_hasil_latih):
            os.mkdir(path_hasil_latih)
        
        for y in master_req_mengambil:
            req_mahasiswa = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-mahasiswa?mahasiswa='+str(y['nim']))
            json_req_mahasiswa = req_mahasiswa.json()
            master_req_mahasiswa = json_req_mahasiswa['master']
            nama_mhs = master_req_mahasiswa[0]['nama']
            
            # buat path mahasiswa menggunakan nama

            path_mahasiswa = os.path.join(path_data_latih, str(nama_mhs))
            if not os.path.exists(path_mahasiswa):
                os.mkdir(path_mahasiswa)
                print(f"Created {path_mahasiswa} = {y['nim']}")

            # download gambar dari firebase
            # akses folder uploads di firebase
            uploads = database.child("uploads").get()

            for i in uploads.each():
                if str(y['nim']) == i.key():
                    for j in i.val():
                        url = i.val()[j]['imageUrl']
                        file_name = i.val()[j]['name']
                        if not os.path.exists(path_mahasiswa+"/"+j+"-"+file_name+".jpg"):
                            urllib.request.urlretrieve(url, path_mahasiswa+"/"+j+"-"+file_name+".jpg") # nama file yg disimpan
                            # download gambar yg telah diupload mahasiswa
                            print("[SUCCESS] download "+file_name+".jpg berhasil")
                        else:
                            print(f"[TERSEDIA] {file_name}.jpg ")

        # cek data di firebase
        if not os.path.exists(path_hasil_latih+"/trained_knn_model.clf"):
            klasifikasi = train(path_kelas+"/data/latih", model_save_path=path_hasil_latih+"/trained_knn_model.clf", n_neighbors=2)
            print(f"Training Kelas {kelas} Selesai")
            # simpan data latih ke firebase
            storage.child("uploads/kelas/latih/"+kelas+"_"+matkul+"/trained_knn_model.clf").put(path_hasil_latih+"/trained_knn_model.clf")
            data = {
                "model" : "trained_knn_model.clf"
            }
            database.child("uploads/kelas/latih/"+kelas+"_"+matkul).child("latih").set(data)
            
        else:
            # simpan data ke firebase jika hosting menggunakan heroku
            storage.child("uploads/kelas/latih/"+kelas+"_"+matkul+"/trained_knn_model.clf").put(path_hasil_latih+"/trained_knn_model.clf")
            data = {
                "model" : "trained_knn_model.clf"
            }
            database.child("uploads/kelas/latih/"+kelas+"_"+matkul).child("latih").set(data)
            print("Data Training Sudah Ada")
            
        waktu = (time.time() - start_time)
        waktu_filter = divmod(waktu, 60)
        menit = waktu_filter[0]
        detik = waktu_filter[1]
        print(menit,'menit',detik,'detik')

        return render_template('index.html', message='Data Latih Selesai')

# buat data 
@app.route('/buat/<kelas>/<matkul>', methods=['GET'])
def buat(kelas, matkul):
    response = []

    # select id kelas
    req_kelas = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-kelas?kelas='+str(kelas))
    json_req_kelas = req_kelas.json()
    master_req_kelas = json_req_kelas['master']
    if not len(master_req_kelas) == 0:
        id_kelas = master_req_kelas[0]['id_kelas']
    else:
        response.append({
            "status":"error - id kelas"
        })
        return jsonify(response)

    # select id matkul
    req_matkul = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-matkul?matkul='+str(matkul))
    json_req_matkul = req_matkul.json()
    master_req_matkul = json_req_matkul['master']
    if not len(master_req_matkul) == 0:
        id_matkul = master_req_matkul[0]['id_matakuliah']
    else:
        response.append({
            "status":"error - id mata kuliah"
        })
        return jsonify(response)

    # select id mengajar
    # revisi, select id mengajar, juga harus pakai id_mata_kuliah, get semester aktif
    req_mengajar = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-dosen?dosen='+str(id_kelas)+'&matkul='+str(id_matkul))
    json_req_mengajar = req_mengajar.json()
    master_req_mengajar = json_req_mengajar['master']
    id_mengajar = master_req_mengajar[0]['id_mengajar']

    # select nim mahasiswa 
    req_mengambil = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-mhs?mhs='+str(id_mengajar))
    json_req_mengambil = req_mengambil.json()
    master_req_mengambil = json_req_mengambil['master']

    # insert data presensi baru
    # select id_presensi & pertemuan from tb_presensi
    # request ini tidak dipakai dulu
    req_pertemuan = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-presensi?presensi='+str(id_mengajar))
    json_req_pertemuan = req_pertemuan.json()
    master_req_pertemuan = json_req_pertemuan['master']

    dir_kelas = "kelas_"+kelas+"_"+matkul
    path_kelas = os.path.join(BASEDIR, dir_kelas)

    if not os.path.exists(path_kelas):
        print(f"[CREATED] {path_kelas}")
        os.mkdir(path_kelas)
    for y in master_req_mengambil:
        req_mahasiswa = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-mahasiswa?mahasiswa='+str(y['nim']))
        json_req_mahasiswa = req_mahasiswa.json()
        master_req_mahasiswa = json_req_mahasiswa['master']
        nama_mhs = master_req_mahasiswa[0]['nama']
        # buat path mahasiswa menggunakan nama
        path_data = os.path.join(path_kelas, "data")
        if not os.path.exists(path_data):
            os.mkdir(path_data)

        path_data_latih = os.path.join(path_data, "latih")
        if not os.path.exists(path_data_latih):
            os.mkdir(path_data_latih)

        path_mahasiswa = os.path.join(path_data_latih, str(nama_mhs))
        if not os.path.exists(path_mahasiswa):
            os.mkdir(path_mahasiswa)
            print(f"Created {path_mahasiswa} = {y['nim']}")

        # download gambar dari firebase
        # akses folder uploads di firebase
        uploads = database.child("uploads").get()

        for i in uploads.each():
            if str(y['nim']) == i.key():
                for j in i.val():
                    url = i.val()[j]['imageUrl']
                    file_name = i.val()[j]['name']
                    if not os.path.exists(path_mahasiswa+"/"+j+"-"+file_name+".jpg"):
                        urllib.request.urlretrieve(url, path_mahasiswa+"/"+j+"-"+file_name+".jpg") # nama file yg disimpan
                        # download gambar yg telah diupload mahasiswa
                        print("[SUCCESS] download "+file_name+".jpg berhasil")
                    else:
                        print(f"[TERSEDIA] {file_name}.jpg ")
    response.append({
        "status":"success"
    })
    # else:
    #     print("[INFO] Data Tersedia")
    #     response.append({
    #         "status":"success"
    #     })
    
    return jsonify(response)

# latih data dan buat model data latih
@app.route('/latih/<kelas>/<matkul>', methods=['GET'])
def latih(kelas, matkul):
    response = []
    dir_kelas = "kelas_"+kelas+"_"+matkul
    path_kelas = os.path.join(BASEDIR, dir_kelas)
    path_hasil = os.path.join(path_kelas, "hasil")
    if not os.path.exists(dir_kelas):
        response.append({
            "status":"Not found "+dir_kelas+" (Buat data dulu)"
        })
        return jsonify(response)
        # os.mkdir(path_hasil)
    
    if not os.path.exists(path_hasil):
        os.mkdir(path_hasil)

    path_hasil_latih = os.path.join(path_hasil, "latih")
    if not os.path.exists(path_hasil_latih):
        os.mkdir(path_hasil_latih)

    path_data = os.path.join(path_kelas, "data")
    if not os.path.exists(path_data):
        os.mkdir(path_data)

    path_data_latih = os.path.join(path_data, "latih")
    if not os.path.exists(path_data_latih):
        os.mkdir(path_data_latih)

    # cek data di firebase
    if not os.path.exists(path_hasil_latih+"/trained_knn_model.clf"):
        klasifikasi = train(path_kelas+"/data/latih", model_save_path=path_hasil_latih+"/trained_knn_model.clf", n_neighbors=2)
        print(f"Training Kelas {kelas} Selesai")
        # simpan data latih ke firebase
        storage.child("uploads/kelas/latih/"+kelas+"_"+matkul+"/trained_knn_model.clf").put(path_hasil_latih+"/trained_knn_model.clf")
        data = {
            "model" : "trained_knn_model.clf"
        }
        database.child("uploads/kelas/latih/"+kelas+"_"+matkul).child("latih").set(data)
        response.append({
            "status":"success"
        })
    else:
        # simpan data ke firebase jika hosting menggunakan heroku
        storage.child("uploads/kelas/latih/"+kelas+"_"+matkul+"/trained_knn_model.clf").put(path_hasil_latih+"/trained_knn_model.clf")
        data = {
            "model" : "trained_knn_model.clf"
        }
        database.child("uploads/kelas/latih/"+kelas+"_"+matkul).child("latih").set(data)
        print("Data Training Sudah Ada")
        response.append({
            "status":"success"
        })

    return jsonify(response)

@app.route('/prediksi/<kelas>/<presensi>/<matkul>', methods=['GET'])
def prediksi(kelas, presensi, matkul):
    start_time = time.time()
    hasil = {}
    response = []
    # direktori tambah matkul setelah kelas, utk tidak redundant
    # cek jika ada direktori yg kosong
    dir_kelas = "kelas_"+kelas+"_"+matkul
    path_kelas = os.path.join(BASEDIR, dir_kelas)
    if not os.path.exists(path_kelas):
        # print(f"[CREATE] {path_kelas}")
        response.append({
            "status":"Not found "+path_kelas+" (Buat data dulu)"
        })
        return jsonify(response)
        # os.mkdir(path_kelas)

    path_data = os.path.join(path_kelas, "data")
    if not os.path.exists(path_data):
        print(f"[CREATE] {path_data}")
        os.mkdir(path_data)

    path_data_latih = os.path.join(path_data, "latih")
    if not os.path.exists(path_data_latih):
        print(f"[CREATE] {path_data_latih}")
        os.mkdir(path_data_latih)

    path_data_uji = os.path.join(path_data, "uji")
    if not os.path.exists(path_data_uji):
        print(f"[CREATE] {path_data_uji}")
        os.mkdir(path_data_uji)

    path_hasil = os.path.join(path_kelas, "hasil")
    if not os.path.exists(path_hasil):
        print(f"[CREATE] {path_hasil}")
        os.mkdir(path_hasil)

    path_hasil_latih = os.path.join(path_hasil, "latih")
    if not os.path.exists(path_hasil_latih):
        print(f"[CREATE] {path_hasil_latih}")
        os.mkdir(path_hasil_latih)
        
    # buat direktori data/uji/id_presensi
    path_data_uji_presensi = os.path.join(path_data_uji, str(presensi))
    if not os.path.exists(path_data_uji_presensi):
        print(f"[CREATE] {path_data_uji_presensi}")
        os.mkdir(path_data_uji_presensi)
    else:
        for files_uji in os.listdir(path_data_uji_presensi):
            full_path = os.path.join(path_data_uji_presensi, files_uji)
            print(f"[HAPUS] Data uji - {full_path}")
            os.remove(full_path)
    
    # buat file json hasil wajah
    path_hasil_uji = os.path.join(path_hasil, "uji")
    if not os.path.exists(path_hasil_uji):
        print(f"[CREATED] {path_hasil_uji}")
        os.mkdir(path_hasil_uji)
    # buat folder hasil/uji/id_presensi
    path_hasil_uji_presensi = os.path.join(path_hasil_uji, str(presensi))
    if not os.path.exists(path_hasil_uji_presensi):
        print(f"[CREATED] {path_hasil_uji_presensi}")
        os.mkdir(path_hasil_uji_presensi)
    else:
        for files in os.listdir(path_hasil_uji_presensi+"/foto"):
            full_path = os.path.join(path_hasil_uji_presensi+"/foto", files)
            print(f"[HAPUS] Hasil uji - {full_path}")
            os.remove(full_path)

    # path_data_prediksi = os.path.join(path_kelas, path_data_uji+"/"+key+".jpg")

    # download semua gambar presensi kelas
    data_kelas = database.child("uploads/kelas/"+presensi).get()
    for data in data_kelas.each():
        key = data.key()
        value = data.val()
        url = value['imageUrl']
        # download gambar semua
        if not os.path.exists(path_data_uji_presensi+"/"+key+".jpg"):
            print(f"[DOWNLOAD] {key}")
            urllib.request.urlretrieve(url, path_data_uji_presensi+"/"+key+".jpg")
        else:
            print(f"[TERSEDIA] {key}.jpg")
    
    # download gambar kelas satu persatu
    # print("[DOWNLOAD] Data Uji"+path_data_prediksi)
    
    # upload_kelas = database.child("uploads/kelas/"+presensi).get()
    # url = upload_kelas.val()[key]['imageUrl']
    # file_name = upload_kelas.val()[key]['name']
    # urllib.request.urlretrieve(url, path_data_prediksi)
    # download data training jika tidak ada (karena sebelumnya di heroku)-heroku menghapus file yg baru dicreate
    if not os.path.exists(path_hasil_latih+"/trained_knn_model.clf"):
        storage.child("uploads/kelas/latih/"+kelas+"_"+matkul+"/trained_knn_model.clf").download(path_hasil_latih+"/trained_knn_model.clf")

    if os.path.exists(f"{path_hasil_uji_presensi}/hasil.txt"): # hapus hasil.txt lama
        os.remove(f"{path_hasil_uji_presensi}/hasil.txt")
        print("[HAPUS] Hapus hasil.txt lama")
    
    for image_file in os.listdir(path_data_uji_presensi):
        full_file_path = os.path.join(path_data_uji_presensi, image_file)

        print("Prediksi gambar {}".format(image_file))

        prediksi = predict(full_file_path, model_path=path_hasil_latih+"/trained_knn_model.clf")
        

        # print hasil di terminal
        # hitung = 1
        for nama, (top, right, bottom, left), dist in prediksi:

            print("[DETECT] Wajah {} : {},{} - {}".format(nama, left, top, dist))
            # get hasil crop wajah
            image_uji = face_recognition.load_image_file(full_file_path)
            image_wajah = image_uji[top:bottom, left:right]
            pil_image = Image.fromarray(image_wajah)
            if not os.path.exists(f"{path_hasil_uji_presensi}/{nama}"):
                os.mkdir(f"{path_hasil_uji_presensi}/{nama}")
            pil_image.save(f"{path_hasil_uji_presensi}/{nama}/{top}-{presensi}.jpg") # simpan hasil crop wajah di direktori
            # simpan di firebase
            nim = 'unknown'
            if nama is not 'unknown':
                req_getnim = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-nim-mahasiswa?mahasiswa='+str(nama)) # NAMA HARUS BENAR
                json_req_getnim = req_getnim.json()
                if (len(json_req_getnim['master']) == 0): # cek jika kosong
                    revisi = nama+"." # kesalahan umum, karena kurang titik di nama
                    req_getnim = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-nim-mahasiswa?mahasiswa='+str(revisi)) # NAMA HARUS BENAR
                    json_req_getnim = req_getnim.json()
                    if (len(json_req_getnim['master']) == 0): # jika salah lagi, return response
                        response.append({
                            "status":"Not found "+revisi
                        })
                        
                        return jsonify(response)
                nim = json_req_getnim['master'][0]['nim']
            upload = storage.child(f"uploads/kelas/hasil/{presensi}/crop/{nim}/{top}-{presensi}.jpg").put(f"{path_hasil_uji_presensi}/{nama}/{top}-{presensi}.jpg")
            get_url = storage.child(f"uploads/kelas/hasil/{presensi}/crop/{nim}/{top}-{presensi}.jpg").get_url(upload['downloadTokens'])
            # get url foto sesuai nama
            get_url_nama = "https://image.flaticon.com/icons/svg/2521/2521768.svg"
            # data_mhs = database.child(f"uploads/{nim}").get()
            # for mhs in data_mhs.each():
            #     value = mhs.val()
            #     get_url_nama = value['imageUrl'] # get url terakhir dari loop
            # get crop gambar
            if nama is not 'unknown':
                for gambar in os.listdir(path_data_latih+"/"+str(nama)):
                    if not os.path.exists(f"{path_data_latih}/{nama}/{nama}.jpg"): # jika sudah ada gambar yg tersimpan, maka lanjut ke folder lain
                        image = face_recognition.load_image_file(path_data_latih+"/"+nama+"/"+gambar)
                        gmbr_face_loc = face_recognition.face_locations(image)
                        if len(gmbr_face_loc) == 1:
                            # print(len(gmbr_face_loc))
                            for face_loc in gmbr_face_loc:
                                topp, right, bottom, left = face_loc

                                face_image = image[topp:bottom, left:right]
                                pil_images = Image.fromarray(face_image)
                                pil_images.save(f"{path_data_latih}/{nama}/{nama}.jpg") # simpan gambar wajah db
                                upload = storage.child(f"uploads/{nim}/{nama}.jpg").put(f"{path_data_latih}/{nama}/{nama}.jpg")
                                get_url_nama = storage.child(f"uploads/{nim}/{nama}.jpg").get_url(upload['downloadTokens'])
                            # kemungkinan bug
                    else:
                        upload = storage.child(f"uploads/{nim}/{nama}.jpg").put(f"{path_data_latih}/{nama}/{nama}.jpg")
                        get_url_nama = storage.child(f"uploads/{nim}/{nama}.jpg").get_url(upload['downloadTokens'])
                        break # jika ada, berhenti loop

            print(f"[UPLOADED] {nim} success")
            # cek hasil.txt
            if not os.path.exists(f"{path_hasil_uji_presensi}/hasil.txt"):
                # data = {}
                # data[] = {
                #     "nama" : nama,
                #     "url" : get_url,
                #     "distance" : dist
                # }
                data = []
                data.append({
                    "nama" : nama,
                    "nim" : nim,
                    "url" : get_url,
                    "url_nama" : get_url_nama,
                    "top" : str(top),
                    "dist" : str(dist)
                })
                with open(f"{path_hasil_uji_presensi}/hasil.txt", 'w') as hasil_file:
                    print("[FIRST] saved data")
                    json.dump(data, hasil_file)
            else:
                # load data dan tambah isi hasil
                # data = {}
                data = []
                with open(f"{path_hasil_uji_presensi}/hasil.txt") as hasil_file:
                    data = json.load(hasil_file)
                    
                # data[] = {
                #     "nama" : nama,
                #     "url" : get_url,
                #     "distance" : dist
                # }
                data.append({
                    "nama" : nama,
                    "nim" : nim,
                    "url" : get_url,
                    "url_nama" : get_url_nama,
                    "top" : str(top),
                    "dist" : str(dist)
                })
                # save data 
                with open(f"{path_hasil_uji_presensi}/hasil.txt", 'w') as hasil_file:
                    print("[EDIT] saved data")
                    json.dump(data, hasil_file)
            # hitung += 1
        
        # with open(path_hasil_uji_presensi+"/hasil.txt", 'w') as hasil_file:
        #     print("[SAVED] file hasil.txt")
        #     json.dump(hasil, hasil_file)
        # simpan di firebase
        data_hasil = {
            "hasil":"hasil.txt"
        }
        database.child("uploads/kelas/hasil/"+presensi).child("hasil").set(data_hasil)
        storage.child("uploads/kelas/hasil/"+presensi+"/hasil.txt").put(path_hasil_uji_presensi+"/hasil.txt")
        print(f"[SUCCESS] upload hasil.txt")
        
        show_prediction_labels_on_image(os.path.join(path_data_uji_presensi, image_file), prediksi, kelas, path_hasil_uji_presensi, presensi)

        waktu = (time.time() - start_time)
        waktu_filter = divmod(waktu, 60)
        menit = waktu_filter[0]
        detik = waktu_filter[1]

        print(f'{menit:.0f} menit {detik:.2f} detik - {image_file}')
    
    file_arr = [] # temp data path file
    for files in os.listdir(path_hasil_uji_presensi+"/foto"):
        file_path = os.path.join(path_hasil_uji_presensi+"/foto", files)
        file_arr.append(file_path)
    # combine hasil prediksi wajah kelas (dimensi harus sama)
    get_url = ""
    if len(file_arr) == 2:
        # img1 = cv2.imread(file_arr[0])
        # img2 = cv2.imread(file_arr[1])
        # vis = np.concatenate((img1, img2), axis=1)
        # cv2.imwrite(path_hasil_uji_presensi+"/foto/hasil.jpg", vis)
        imgA = Image.open(file_arr[0])
        imgB = Image.open(file_arr[1])
        combine = Image.new('RGB', (imgA.width + imgB.width, imgA.height))
        combine.paste(imgA, (0, 0))
        combine.paste(imgB, (imgA.width, 0))
        combine.save(path_hasil_uji_presensi+"/foto/hasil.jpg")
        # save firebase
        upload = storage.child("uploads/kelas/hasil/"+presensi+"/foto/hasil.jpg").put(path_hasil_uji_presensi+"/foto/hasil.jpg")
        # get url gambar, simpan di url_gambar.txt
        get_url = storage.child("uploads/kelas/hasil/"+presensi+"/foto/hasil.jpg").get_url(upload['downloadTokens'])
    else:
        # save firebase
        upload = storage.child("uploads/kelas/hasil/"+presensi+"/foto/hasil.jpg").put(file_arr[0])
        # get url gambar, simpan di url_gambar.txt
        get_url = storage.child("uploads/kelas/hasil/"+presensi+"/foto/hasil.jpg").get_url(upload['downloadTokens'])
    
    # input data url ke database top presence
    data_presensi = {
        'id_presensi' : str(presensi),
        'wajah' : str(get_url)
    }
    req_presensi = requests.post('https://topapp.id/top-presence/api/v1/presensi-detail/url-presensi-wajah', data=data_presensi)
    response_req = req_presensi.json()
    # response_req = '200'
    # if response_req == '200':
    if response_req['code'] == '200':
        print("[STATUS] Success Wajah Kelas")
    else:
        print("[STATUS] Failed Wajah Kelas")
    
    waktu = (time.time() - start_time)
    waktu_filter = divmod(waktu, 60)
    menit = waktu_filter[0]
    detik = waktu_filter[1]

    print(f'{menit:.0f} menit {detik:.2f} detik')

    response.append({
        "status":"success"
    })
    
    return jsonify(response)

# route hasil presensi
@app.route('/hasil/<kelas>/<presensi>/<matkul>', methods=['GET'])
def hasil_presensi(kelas, presensi, matkul):
    data = []

    dir_kelas = "kelas_"+kelas+"_"+matkul
    path_kelas = os.path.join(BASEDIR, dir_kelas)
    if not os.path.exists(path_kelas):
        data.append({
            "status":"Not found "+path_kelas+" (Buat data dulu)"
        })
        return jsonify(data)
    #     print(f"[CREATED] {path_kelas}")
    #     os.mkdir(path_kelas)
    
    path_hasil = os.path.join(path_kelas, "hasil")
    # if not os.path.exists(path_hasil):
    #     print(f"[CREATED] {path_hasil}")
    #     os.mkdir(path_hasil)
    
    path_hasil_uji = os.path.join(path_hasil, "uji")
    # if not os.path.exists(path_hasil_uji):
    #     print(f"[CREATED] {path_hasil_uji}")
    #     os.mkdir(path_hasil_uji)

    path_hasil_uji_presensi = os.path.join(path_hasil_uji, str(presensi))
    # if not os.path.exists(path_hasil_uji_presensi):
    #     print(f"[CREATED] {path_hasil_uji_presensi}")
    #     os.mkdir(path_hasil_uji_presensi)
    
    if not os.path.exists(path_hasil_uji_presensi+"/hasil.txt"):
        # download dari firebase
        # storage.child("uploads/kelas/hasil/"+presensi+"/hasil.txt").download(path_hasil_uji_presensi+"/hasil.txt")
        print("[INFO] empty hasil")
        data.append({
            "status":"Data hasil Kosong"
        })
    else:
        with open(path_hasil_uji_presensi+"/hasil.txt") as hasil_file:
            data = json.load(hasil_file)
        
        # koreksi hasil presensi by sistem
        nama = []
        # 1. hapus unknown
        for i in data:
            nama.append(i['nama']) # get nama
            if i['nama'] == 'unknown':
                data.remove(i)
        # 2. hapus data sama dengan membandingkan treshold (antara 2 data)
        # get duplicate
        duplicate = [item for item, count in collections.Counter(nama).items() if count > 1]
        dist_dups = [] # get ditance duplicate
        for v in range(len(duplicate)):
            for j in data:
                if j['nama'] == duplicate[v]: # testing sebelum sidang
                    dist_dups.append(float(j['dist']))
            for k in data:
                if k['dist'] == str(max(dist_dups)):
                    data.remove(k)
                    print('remove_duplicate:', k['nama'])
            dist_dups = []

        # 3. hapus data dengan jumlah treshload lebih dari 0.5
        
        # save data 
        with open(path_hasil_uji_presensi+"/hasil.txt", 'w') as hasil_file:
            print("[NEW] saved new data")
            json.dump(data, hasil_file)
        storage.child("uploads/kelas/hasil/"+presensi+"/hasil.txt").put(path_hasil_uji_presensi+"/hasil.txt") # update data firebase

    return jsonify(data)

# route isi presensi
@app.route('/isipresensi/<kelas>/<presensi>/<matkul>', methods=['GET'])
def isi_presensi(kelas, presensi, matkul):
    response_hasil = []
    
    dir_kelas = "kelas_"+kelas+"_"+matkul
    path_kelas = os.path.join(BASEDIR, dir_kelas)
    if not os.path.exists(path_kelas):
        response_hasil.append({
            "status":"Not found "+path_kelas+" (Buat data dulu)"
        })
        return jsonify(response_hasil)
    #     print(f"[CREATED] {path_kelas}")
    #     os.mkdir(path_kelas)
    
    path_hasil = os.path.join(path_kelas, "hasil")
    # if not os.path.exists(path_hasil):
    #     print(f"[CREATED] {path_hasil}")
    #     os.mkdir(path_hasil)
    
    path_hasil_uji = os.path.join(path_hasil, "uji")
    # if not os.path.exists(path_hasil_uji):
    #     print(f"[CREATED] {path_hasil_uji}")
    #     os.mkdir(path_hasil_uji)

    path_hasil_uji_presensi = os.path.join(path_hasil_uji, str(presensi))
    # if not os.path.exists(path_hasil_uji_presensi):
    #     print(f"[CREATED] {path_hasil_uji_presensi}")
    #     os.mkdir(path_hasil_uji_presensi)
    
    if not os.path.exists(path_hasil_uji_presensi+"/hasil.txt"):
        # download dari firebase
        storage.child("uploads/kelas/hasil/"+presensi+"/hasil.txt").download(path_hasil_uji_presensi+"/hasil.txt")
    
    # loop di hasil.txt dan isi presensi
    data = []
    with open(path_hasil_uji_presensi+"/hasil.txt") as hasil_file:
        data = json.load(hasil_file)
    
    for x in data:
        data_presensi = {
            'id_presensi' : str(presensi),
            'nim' : str(x['nim']),
            'wajah' : str(x['url'])
        }
        # cek jika sudah terisi hadir
        req_cek_presensi = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-mhs-detail-presensi?id_p='+str(presensi)+'&mhs='+str(x['nim']))
        json_req_cek_presensi = req_cek_presensi.json()
        master_req_cek_presensi = json_req_cek_presensi['master']
        if master_req_cek_presensi[0]['status'] == 'Tidak Hadir':
            print(f"[INFO] Mengisi Presensi: {x['nama']}")
            req_presensi = requests.post('https://topapp.id/top-presence/api/v1/presensi-detail/terima-presensi-wajah', data=data_presensi)
            response = req_presensi.json()
            # response = '200'
            # if response == '200':
            if response['code'] == '200':
                print("[STATUS] Success")
                response_hasil.append({
                    "status":"success"
                })
            else:
                print("[STATUS] Failed")
                response_hasil.append({
                    "status":"failed"
                })
        else:
            print(f"Presensi {x['nama']} sudah terisi")
            # isi url hasil prediksi

    return jsonify(response_hasil)

# route validasi kesalahan prediksi (hapus isi yg tidak valid)/ edit hasil.txt
@app.route('/validasi/<kelas>/<presensi>/<matkul>/<top>', methods=['GET'])
def validasi(kelas, presensi, matkul, top):
    data = []

    dir_kelas = "kelas_"+kelas+"_"+matkul
    path_kelas = os.path.join(BASEDIR, dir_kelas)
    if not os.path.exists(path_kelas):
        data.append({
            "status":"Not found "+path_kelas+" (Buat data dulu)"
        })
        return jsonify(data)
    #     print(f"[CREATED] {path_kelas}")
    #     os.mkdir(path_kelas)
    
    path_hasil = os.path.join(path_kelas, "hasil")
    # if not os.path.exists(path_hasil):
    #     print(f"[CREATED] {path_hasil}")
    #     os.mkdir(path_hasil)
    
    path_hasil_uji = os.path.join(path_hasil, "uji")
    # if not os.path.exists(path_hasil_uji):
    #     print(f"[CREATED] {path_hasil_uji}")
    #     os.mkdir(path_hasil_uji)
    
    path_hasil_uji_presensi = os.path.join(path_hasil_uji, str(presensi))
    # if not os.path.exists(path_hasil_uji_presensi):
    #     print(f"[CREATED] {path_hasil_uji_presensi}")
    #     os.mkdir(path_hasil_uji_presensi)
    
    if not os.path.exists(path_hasil_uji_presensi+"/hasil.txt"):
        # download dari firebase
        storage.child("uploads/kelas/hasil/"+presensi+"/hasil.txt").download(path_hasil_uji_presensi+"/hasil.txt")
    
    # edit isi hasil.txt
    
    with open(path_hasil_uji_presensi+"/hasil.txt") as hasil_file:
        data = json.load(hasil_file)
    
    for x in data:
        if x['top'] == top:
            print(f"[INFO] Data {x['nama']} dihapus")
            data.remove(x)
            break # end loop jika sudah ketemu, karena validasi satu satu
        
    # save data 
    with open(path_hasil_uji_presensi+"/hasil.txt", 'w') as hasil_file:
        print("[NEW] saved new data")
        json.dump(data, hasil_file)
    storage.child("uploads/kelas/hasil/"+presensi+"/hasil.txt").put(path_hasil_uji_presensi+"/hasil.txt") # update data firebase
    
    # print(data[1]['nim'] == nim)

    return jsonify(data)

# route get hasil foto presensi
@app.route('/foto/<presensi>', methods=['GET'])
def hasil_uji_foto(presensi):
    data = []
    data_foto = database.child(f"uploads/kelas/hasil/{presensi}/foto").get()
    for x in data_foto.each():
        value = x.val()
        # print(value['url'])
        data.append({
            "url":value['url']
        })
        
    return jsonify(data)
    
# Single Face 1 to 1 ==============================================

# route single data
@app.route('/single_auth/<nim>/<presensi>', methods=['GET'])
def single_auth(nim, presensi):
    response = []

    dir_single = os.path.join(BASEDIR, "single") # path single face
    if not os.path.exists(dir_single):
        os.mkdir(dir_single)

    path_nim = os.path.join(dir_single, str(nim)) # single/nim
    if not os.path.exists(path_nim):
        os.mkdir(path_nim)

    path_data = os.path.join(path_nim, "data") # single/nim/data
    if not os.path.exists(path_data):
        os.mkdir(path_data)

    path_hasil = os.path.join(path_nim, "hasil") # single/nim/hasil
    if not os.path.exists(path_hasil):
        os.mkdir(path_hasil)
    
    path_data_latih = os.path.join(path_data, "latih") # single/nim/data/latih
    if not os.path.exists(path_data_latih):
        os.mkdir(path_data_latih)
    
    path_data_uji = os.path.join(path_data, "uji") # single/nim/data/uji
    if not os.path.exists(path_data_uji):
        os.mkdir(path_data_uji)
    
    path_hasil_latih = os.path.join(path_hasil, "latih") # single/nim/hasil/latih
    if not os.path.exists(path_hasil_latih):
        os.mkdir(path_hasil_latih)
    
    path_hasil_uji = os.path.join(path_hasil, "uji") # single/nim/hasil/uji
    if not os.path.exists(path_hasil_uji):
        os.mkdir(path_hasil_uji)
    
    path_data_uji_presensi = os.path.join(path_data_uji, str(presensi)) # single/nim/data/uji/presensi
    if not os.path.exists(path_data_uji_presensi):
        os.mkdir(path_data_uji_presensi)
    else: # jika tersedia data uji, hapus dulu, lalu buat ulang supaya isinya 1 file
        for files in os.listdir(path_data_uji_presensi):
            full_path = os.path.join(path_data_uji_presensi, files)
            print(f"[INFO] Hapus {full_path}")
            os.remove(full_path)
    
    path_hasil_uji_presensi = os.path.join(path_hasil_uji, str(presensi)) # single/nim/hasil/uji/presensi
    if not os.path.exists(path_hasil_uji_presensi):
        os.mkdir(path_hasil_uji_presensi)

    # download data latih berdasarkan nim

    req_mahasiswa = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-mahasiswa?mahasiswa='+str(nim))
    json_req_mahasiswa = req_mahasiswa.json()
    master_req_mahasiswa = json_req_mahasiswa['master']
    nama_mhs = master_req_mahasiswa[0]['nama'] # nama berdasarkan nim

    path_data_latih_nama = os.path.join(path_data_latih, nama_mhs) # single/nim/data/latih/nama
    if not os.path.exists(path_data_latih_nama):
        os.mkdir(path_data_latih_nama)
    
    uploads = database.child("uploads").get()

    for i in uploads.each():
        if str(nim) == i.key():
            for j in i.val():
                url = i.val()[j]['imageUrl']
                file_name = i.val()[j]['name']
                if not os.path.exists(path_data_latih_nama+"/"+j+"-"+file_name+".jpg"):
                    urllib.request.urlretrieve(url, path_data_latih_nama+"/"+j+"-"+file_name+".jpg") # nama file yg disimpan
                    # download gambar yg telah diupload mahasiswa
                    print("[SUCCESS] download "+file_name+".jpg berhasil")
                else:
                    print(f"[TERSEDIA] {file_name}.jpg ")

    # latih dan save model 

    if not os.path.exists(path_hasil_latih+"/trained_knn_model.clf"):
        klasifikasi = train(path_data_latih, model_save_path=path_hasil_latih+"/trained_knn_model.clf", n_neighbors=2)
        print(f"Training {nim} Selesai")
        
    else:
        print("Data Training Sudah Ada")
        
    # prediksi dan save hasil
    # data uji/presensi, isinya satu file gambar saja utk single auth

    data_single = database.child("uploads/single/"+presensi+"/"+nim).get() # download data uji
    for data in data_single.each():
        key = data.key()
        value = data.val()
        url = value['imageUrl']
        # download gambar semua
        if not os.path.exists(path_data_uji_presensi+"/"+key+".jpg"):
            print(f"[DOWNLOAD] {key}")
            urllib.request.urlretrieve(url, path_data_uji_presensi+"/"+key+".jpg") # download foto wajah utk diprediksi
        else:
            print(f"[TERSEDIA] {key}.jpg")

    # hapus wajah prediksi di firebase
    database.child(f"uploads/single/{presensi}").child(nim).remove()
    print("[INFO] Data Prediksi Dihapus")
    
    for image_file in os.listdir(path_data_uji_presensi): # loop dalam folder utk diprediksi
        full_file_path = os.path.join(path_data_uji_presensi, image_file)

        print("Looking for faces in {}".format(image_file))

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(full_file_path, model_path=path_hasil_latih+"/trained_knn_model.clf")

        print(f"[PREDICT] {predictions}")
        # Print results on the console
        for nama, (top, right, bottom, left), dist in predictions: # cek nama prediksi, jika kosong
            # isi presensi disini
            # upload crop wajah prediksi
            X_img = face_recognition.load_image_file(full_file_path)
            face_image = X_img[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            pil_image.save(f'{path_hasil_uji_presensi}/{top}-{presensi}.jpg') # crop wajah disimpan

            print("- [Found] {} at ({}, {}) in {}".format(nama, left, top, dist))
            # upload ke firebase
            upload = storage.child(f"uploads/kelas/hasil/{presensi}/crop/{nim}/{top}-{presensi}.jpg").put(f"{path_hasil_uji_presensi}/{top}-{presensi}.jpg")
            get_url = storage.child(f"uploads/kelas/hasil/{presensi}/crop/{nim}/{top}-{presensi}.jpg").get_url(upload['downloadTokens'])

            get_url_nama = ""
            # loop dalam nama mahasiswa , bukan mahasiswa prediksi
            for gambar in os.listdir(path_data_latih+"/"+str(nama_mhs)): # jika prediksi salah, maka loop dalam folder nama kemungkinan error
                if not os.path.exists(f"{path_data_latih_nama}/{nama_mhs}.jpg"): # jika sudah ada gambar yg tersimpan, maka lanjut ke folder lain
                    image = face_recognition.load_image_file(path_data_latih+"/"+nama_mhs+"/"+gambar)
                    gmbr_face_loc = face_recognition.face_locations(image)
                    if len(gmbr_face_loc) == 1:
                        # print(len(gmbr_face_loc))
                        for face_loc in gmbr_face_loc:
                            top, right, bottom, left = face_loc

                            face_image = image[top:bottom, left:right]
                            pil_images = Image.fromarray(face_image)
                            pil_images.save(f"{path_data_latih_nama}/{nama_mhs}.jpg") # simpan gambar wajah db
                            upload = storage.child(f"uploads/{nim}/{nama_mhs}.jpg").put(f"{path_data_latih_nama}/{nama_mhs}.jpg")
                            get_url_nama = storage.child(f"uploads/{nim}/{nama_mhs}.jpg").get_url(upload['downloadTokens'])
                else:
                    print(f"[INFO] crop wajah {nama_mhs} tersedia")
                    upload = storage.child(f"uploads/{nim}/{nama_mhs}.jpg").put(f"{path_data_latih_nama}/{nama_mhs}.jpg")
                    get_url_nama = storage.child(f"uploads/{nim}/{nama_mhs}.jpg").get_url(upload['downloadTokens'])
                    break # loop berhenti jika crop wajah tersedia
            
            if os.path.exists(f"{path_hasil_uji_presensi}/hasil.txt"): # jika hasil tersedia, hapus dulu
                print(f"[HAPUS] {path_hasil_uji_presensi}/hasil.txt")
                os.remove(path_hasil_uji_presensi+"/hasil.txt")
                
            data = []
            data.append({
                "nama" : nama,
                "nim" : nim,
                "url" : get_url,
                "url_nama" : get_url_nama
            })
            with open(f"{path_hasil_uji_presensi}/hasil.txt", 'w') as hasil_file:
                print("[SAVED] saved data")
                json.dump(data, hasil_file)
            storage.child("uploads/single/"+presensi+"/hasil/"+nim+"/hasil.txt").put(path_hasil_uji_presensi+"/hasil.txt")
            print(f"[SUCCESS] upload hasil.txt")
        
        # Display results overlaid on an image
        show_prediction_labels_on_image(os.path.join(path_data_uji_presensi, image_file), predictions, None, path_hasil_uji_presensi, presensi)

    # tampilkan hasil prediksi 
    if not os.path.exists(path_hasil_uji_presensi+"/hasil.txt"):
        print("[INFO] empty hasil") # cek jika kosong
    else:
        with open(path_hasil_uji_presensi+"/hasil.txt") as hasil_file:
            response = json.load(hasil_file)

    return jsonify(response)

# route konfirmasi presensi
@app.route('/konfirmasi_auth/<nim>/<presensi>/<key>', methods=['GET'])
def konfirmasi_auth(nim, presensi, key):
    response = []

    dir_single = os.path.join(BASEDIR, "single_v2") # path single face
    # if not os.path.exists(dir_single):
    #     os.mkdir(dir_single)

    # path_nim = os.path.join(dir_single, str(nim)) # single
    # if not os.path.exists(path_nim):
    #     os.mkdir(path_nim)

    path_hasil = os.path.join(dir_single, "hasil") # single/hasil
    # if not os.path.exists(path_hasil):
    #     os.mkdir(path_hasil)

    path_hasil_uji = os.path.join(path_hasil, "uji") # single/hasil/uji
    # if not os.path.exists(path_hasil_uji):
    #     os.mkdir(path_hasil_uji)
    
    path_hasil_uji_presensi = os.path.join(path_hasil_uji, str(presensi)) # single/hasil/uji/presensi
    # if not os.path.exists(path_hasil_uji_presensi):
    #     os.mkdir(path_hasil_uji_presensi)

    path_hasil_uji_p_key = os.path.join(path_hasil_uji_presensi, str(key)) # single/hasil/uji/presensi/key


    data = []
    with open(path_hasil_uji_p_key+"/hasil.txt") as hasil:
        data = json.load(hasil)

    data_presensi = {
        'id_presensi' : str(presensi),
        'nim' : str(nim),
        'wajah' : str(data[0]['url'])
    }

    # cek jika data unknown
    if nim != 'unknown':
        # cek jika sudah Hadir
        req_cek_presensi = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-mhs-detail-presensi?id_p='+str(presensi)+'&mhs='+str(nim))
        json_req_cek_presensi = req_cek_presensi.json()
        master_req_cek_presensi = json_req_cek_presensi['master']
        if master_req_cek_presensi:
            if master_req_cek_presensi[0]['status'] == 'Tidak Hadir':
                req_presensi = requests.post('https://topapp.id/top-presence/api/v1/presensi-detail/terima-presensi-wajah', data=data_presensi)
                response_req = req_presensi.json()
                # response_req = '200'
                # if response_req == '200':
                if response_req['code'] == '200':
                    print("[STATUS] Success isi")
                    response.append({
                        "status":"Berhasil Isi Presensi - "+response_req['code']
                    })
                    # hapus file hasil.txt dan data uji setelah konfirmasi
                    if os.path.exists(path_hasil_uji_p_key+"/hasil.txt"):
                        os.remove(path_hasil_uji_p_key+"/hasil.txt")
                    database.child(f"uploads/single/{presensi}").child(nim).remove()
                else:
                    print("[STATUS] Failed")
                    response.append({
                        "status":"Gagal Isi Presensi - "+response_req['code']
                    })
            else:
                print("[STATUS] Sudah Terisi")
                if os.path.exists(path_hasil_uji_p_key+"/hasil.txt"):
                    os.remove(path_hasil_uji_p_key+"/hasil.txt")
                database.child(f"uploads/single/{presensi}").child(nim).remove()
                response.append({
                    "status":"Presensi Sudah Terisi"
                })
        else: # jika tidak ada nim dalam presensi atau mhs bukan di kelas tsb
            response.append({
                "status":"Mahasiswa kelas lain"
            })
    else:
        response.append({
            "status":"Data Tidak dikenali"
        })

    return jsonify(response)


# Single Auth Face 1 to M
@app.route('/single_auth2/<presensi>/<key>', methods=['GET'])
def single_auth2(presensi, key):
    response = []
    url_foto = "kosong"

    dir_single2 = os.path.join(BASEDIR, "single_v2") # path single face
    if not os.path.exists(dir_single2):
        os.mkdir(dir_single2)
    
    # path_spresensi = os.path.join(dir_single2, str(presensi))
    # if not os.path.exists(path_spresensi):
    #     os.mkdir(path_spresensi)
    #     print("- Create",path_spresensi)
    
    path_data = os.path.join(dir_single2, "data") # single2/data
    if not os.path.exists(path_data):
        os.mkdir(path_data)

    path_hasil = os.path.join(dir_single2, "hasil") # single2/hasil
    if not os.path.exists(path_hasil):
        os.mkdir(path_hasil)
    
    path_data_latih = os.path.join(path_data, "latih") # single2/data/latih
    if not os.path.exists(path_data_latih):
        os.mkdir(path_data_latih)
    
    path_data_uji = os.path.join(path_data, "uji") # single2/data/uji
    if not os.path.exists(path_data_uji):
        os.mkdir(path_data_uji)
    
    path_hasil_latih = os.path.join(path_hasil, "latih") # single2/hasil/latih
    if not os.path.exists(path_hasil_latih):
        os.mkdir(path_hasil_latih)
    
    path_hasil_uji = os.path.join(path_hasil, "uji") # single2/hasil/uji
    if not os.path.exists(path_hasil_uji):
        os.mkdir(path_hasil_uji)
    
    path_data_uji_presensi = os.path.join(path_data_uji, str(presensi)) # single2/data/uji/presensi
    if not os.path.exists(path_data_uji_presensi):
        os.mkdir(path_data_uji_presensi)
    
    path_hasil_uji_presensi = os.path.join(path_hasil_uji, str(presensi)) # single2/hasil/uji/presensi
    if not os.path.exists(path_hasil_uji_presensi):
        os.mkdir(path_hasil_uji_presensi)

    path_data_uji_p_key = os.path.join(path_data_uji_presensi, str(key)) # single2/data/uji/presensi/key
    if not os.path.exists(path_data_uji_p_key):
        os.mkdir(path_data_uji_p_key)
    else: # hapus data lama, utk prediksi satu foto uji saja
        for files in os.listdir(path_data_uji_p_key):
            full_path = os.path.join(path_data_uji_p_key, files)
            print(f"[INFO] Hapus {full_path}")
            os.remove(full_path)
    
    path_hasil_uji_p_key = os.path.join(path_hasil_uji_presensi, str(key)) # single2/hasil/uji/presensi/key
    if not os.path.exists(path_hasil_uji_p_key):
        os.mkdir(path_hasil_uji_p_key)

    # latih dan save model 

    if not os.path.exists(path_hasil_latih+"/trained_knn_model.clf"):
        # data = []
        # data.append({
        #     "nama" : nama,
        #     "nim" : nim,
        #     "url" : get_url,
        #     "url_nama" : get_url_nama
        # })
        # return jsonify(data)

        # download data latih dan train beda route
        # get nim from firebase
        data_mhs = database.child("uploads").get()
        for x in data_mhs.each():
            nim_f = x.key()
            if len(nim_f) == 11:
                req_mahasiswa = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-mahasiswa?mahasiswa='+str(nim_f))
                json_req_mahasiswa = req_mahasiswa.json()
                master_req_mahasiswa = json_req_mahasiswa['master']
                nama_mhs = master_req_mahasiswa[0]['nama'] # nama berdasarkan nim

                path_data_latih_nama = os.path.join(path_data_latih, nama_mhs) # single/data/latih/nama
                if not os.path.exists(path_data_latih_nama):
                    os.mkdir(path_data_latih_nama)

                # download foto dari firebase
                for i in x.val():
                    url = x.val()[i]['imageUrl']
                    file_name = x.val()[i]['name']
                    if not os.path.exists(path_data_latih_nama+"/"+i+"-"+file_name+".jpg"):
                        urllib.request.urlretrieve(url, path_data_latih_nama+"/"+i+"-"+file_name+".jpg") # nama file yg disimpan
                        # download gambar yg telah diupload mahasiswa
                        print("[SUCCESS] download "+file_name+".jpg berhasil")
                    else:
                        print(f"[TERSEDIA] {file_name}.jpg ")

        klasifikasi = train(path_data_latih, model_save_path=path_hasil_latih+"/trained_knn_model.clf", n_neighbors=2)
        print("Data Training Selesai dibuat")
        return jsonify("Data Training Selesai")
    else:
        print("Data Training Sudah Ada")
        
    # prediksi dan save hasil
    # data uji/presensi, isinya satu file gambar saja utk single auth
    data_single = database.child("uploads/single/"+presensi+"/"+key).get() # download data uji
    url = data_single.val()['imageUrl']
    # download gambar semua
    if not os.path.exists(path_data_uji_p_key+"/"+key+".jpg"):
        print(f"[DOWNLOAD] {key}")
        urllib.request.urlretrieve(url, path_data_uji_p_key+"/"+key+".jpg") # download foto wajah utk diprediksi
    else:
        print(f"[TERSEDIA] {key}.jpg")

    # hapus wajah prediksi/uji di firebase
    database.child(f"uploads/single/{presensi}").remove()
    print("[INFO] Data Prediksi/Uji Dihapus",key)
    
    for image_file in os.listdir(path_data_uji_p_key): # loop dalam folder utk diprediksi
        full_file_path = os.path.join(path_data_uji_p_key, image_file)

        print("Looking for faces in {}".format(image_file))

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(full_file_path, model_path=path_hasil_latih+"/trained_knn_model.clf")

        print(f"[PREDICT] {predictions}")
        # Print results on the console
        for nama, (top, right, bottom, left), dist in predictions: # cek nama prediksi, jika kosong
            # isi presensi disini
            # upload crop wajah prediksi
            X_img = face_recognition.load_image_file(full_file_path)
            face_image = X_img[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            pil_image.save(f'{path_hasil_uji_p_key}/{top}-{presensi}.jpg') # crop wajah disimpan

            print("- [Found] {} at ({}, {}) in {}".format(nama, left, top, dist))
            # get nim
            nim = 'unknown'
            if nama is not 'unknown':
                req_getnim = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-nim-mahasiswa?mahasiswa='+str(nama)) # NAMA HARUS BENAR
                json_req_getnim = req_getnim.json()
                if (len(json_req_getnim['master']) == 0): # cek jika kosong
                    revisi = nama+"." # kesalahan umum, karena kurang titik di nama
                    req_getnim = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-nim-mahasiswa?mahasiswa='+str(revisi)) # NAMA HARUS BENAR
                    json_req_getnim = req_getnim.json()
                    if (len(json_req_getnim['master']) == 0): # jika salah lagi, return response
                        response.append({
                            "status":"Not found "+revisi
                        })
                        
                        return jsonify(response)
                nim = json_req_getnim['master'][0]['nim']
            # upload ke firebase
            upload = storage.child(f"uploads/kelas/hasil/{presensi}/crop/{nim}/{top}-{presensi}.jpg").put(f"{path_hasil_uji_p_key}/{top}-{presensi}.jpg")
            get_url = storage.child(f"uploads/kelas/hasil/{presensi}/crop/{nim}/{top}-{presensi}.jpg").get_url(upload['downloadTokens'])

            get_url_nama = "https://image.flaticon.com/icons/svg/2521/2521768.svg"
            # loop dalam nama mahasiswa , bukan mahasiswa prediksi
            if nama is not 'unknown':
                for gambar in os.listdir(path_data_latih+"/"+str(nama)): # jika prediksi salah, maka loop dalam folder nama kemungkinan error
                    path_data_latih_nama = os.path.join(path_data_latih, nama)
                    if not os.path.exists(f"{path_data_latih_nama}/{nama}.jpg"): # jika sudah ada gambar yg tersimpan, maka lanjut ke folder lain
                        image = face_recognition.load_image_file(path_data_latih+"/"+nama+"/"+gambar)
                        gmbr_face_loc = face_recognition.face_locations(image)
                        if len(gmbr_face_loc) == 1:
                            # print(len(gmbr_face_loc))
                            for face_loc in gmbr_face_loc:
                                top, right, bottom, left = face_loc

                                face_image = image[top:bottom, left:right]
                                pil_images = Image.fromarray(face_image)
                                pil_images.save(f"{path_data_latih_nama}/{nama}.jpg") # simpan gambar wajah db
                                upload = storage.child(f"uploads/{nim}/{nama}.jpg").put(f"{path_data_latih_nama}/{nama}.jpg")
                                get_url_nama = storage.child(f"uploads/{nim}/{nama}.jpg").get_url(upload['downloadTokens'])
                    else:
                        print(f"[INFO] crop wajah {nama} tersedia")
                        upload = storage.child(f"uploads/{nim}/{nama}.jpg").put(f"{path_data_latih_nama}/{nama}.jpg")
                        get_url_nama = storage.child(f"uploads/{nim}/{nama}.jpg").get_url(upload['downloadTokens'])
                        break # loop berhenti jika crop wajah tersedia
            
            if os.path.exists(f"{path_hasil_uji_p_key}/hasil.txt"): # jika hasil tersedia, hapus dulu
                print(f"[HAPUS-LAMA] {path_hasil_uji_p_key}/hasil.txt")
                os.remove(path_hasil_uji_p_key+"/hasil.txt")
                
            data = []
            data.append({
                "nama" : nama,
                "nim" : nim,
                "url" : get_url,
                "url_nama" : get_url_nama,
                "key" : key
            })
            with open(f"{path_hasil_uji_p_key}/hasil.txt", 'w') as hasil_file:
                print("[SAVED] saved data")
                json.dump(data, hasil_file)
            storage.child("uploads/single/"+presensi+"/hasil/"+nim+"/hasil.txt").put(path_hasil_uji_p_key+"/hasil.txt")
            print(f"[SUCCESS] upload hasil.txt")
        
        # Display results overlaid on an image
        show_prediction_labels_on_image(os.path.join(path_data_uji_p_key, image_file), predictions, None, path_hasil_uji_p_key, presensi)

    # tampilkan hasil prediksi 
    if not os.path.exists(path_hasil_uji_p_key+"/hasil.txt"):
        print("[INFO] empty hasil") # cek jika kosong / tidak terdeteksi
        for image_file in os.listdir(path_data_uji_p_key): # loop dalam folder utk diupload ke firebase
            full_file_path = os.path.join(path_data_uji_p_key, image_file)
            unix = time.time()
            # upload ke firebase
            upload = storage.child(f"uploads/kelas/hasil/{presensi}/notdetected/{unix}.jpg").put(full_file_path)
            url_notdetected = storage.child(f"uploads/kelas/hasil/{presensi}/notdetected/{unix}.jpg").get_url(upload['downloadTokens'])

        response.append({
            "nama" : "notdetected",
            "nim" : "nim",
            "url" : url_notdetected,
            "url_nama" : "get_url_nama",
            "key" : "key"
        })
        
    else:
        with open(path_hasil_uji_p_key+"/hasil.txt") as hasil_file:
            response = json.load(hasil_file)

    return jsonify(response)

# Run Server
if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000) # production topappvps
    app.run(port=5000, debug=True) # development