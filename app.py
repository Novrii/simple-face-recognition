from flask import Flask, request, jsonify
import os
import calendar
import datetime
import cv2
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

# Init app
app = Flask(__name__)

# Init Firebase
config = {
    "apiKey": "AIzaSyAzqVM2rLXpABaYkZee337wfYQnnQPFPFU",
    "authDomain": "fir-upload-example-f26b6.firebaseapp.com",
    "databaseURL": "https://fir-upload-example-f26b6.firebaseio.com",
    "projectId": "fir-upload-example-f26b6",
    "storageBucket": "fir-upload-example-f26b6.appspot.com",
    "messagingSenderId": "165280253599",
    "appId": "1:165280253599:web:da0a9a30ab2155e94f44aa"
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
database = firebase.database()

# Init directory
BASEDIR = os.path.abspath(os.path.dirname(__file__))

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# fungsi rescale image 
def scale_down(scale, img):
    width = int(img.shape[1] * scale/100)
    height = int(img.shape[0] * scale/100)
    dim = (width, height)
    new_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return new_image

# fungsi cek_size
def cek_size(size, img, path_name, file_name):
    if size > 100000 and size < 200000:
        scale_percent = 70
        rescale = scale_down(scale_percent, img)
        path_save = os.path.join(path_name, file_name)
        cv2.imwrite(path_save, rescale)
        new_size = os.path.getsize(path_save)
        print(f"New Size: {new_size/1024} (100)")
    elif size > 200000 and size < 300000:
        scale_percent = 60
        rescale = scale_down(scale_percent, img)
        path_save = os.path.join(path_name, file_name)
        cv2.imwrite(path_save, rescale)
        new_size = os.path.getsize(path_save)
        print(f"New Size: {new_size/1024} (200)")
    elif size > 300000 and size < 400000:
        scale_percent = 50
        rescale = scale_down(scale_percent, img)
        path_save = os.path.join(path_name, file_name)
        cv2.imwrite(path_save, rescale)
        new_size = os.path.getsize(path_save)
        print(f"New Size: {new_size/1024} (300)")
    elif size > 400000 and size < 500000:
        scale_percent = 40
        rescale = scale_down(scale_percent, img)
        path_save = os.path.join(path_name, file_name)
        cv2.imwrite(path_save, rescale)
        new_size = os.path.getsize(path_save)
        print(f"New Size: {new_size/1024} (400)")
    elif size > 500000:
        scale_percent = 30
        rescale = scale_down(scale_percent, img)
        path_save = os.path.join(path_name, file_name)
        cv2.imwrite(path_save, rescale)
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
            img = cv2.imread(img_path)
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
    X_face_locations = face_recognition.face_locations(X_img) # default model="hog", pilihan lain "cnn"

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

    # Simpan crop gambar utk validasi di Android (Upload ke firebas, get url image)
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
    pil_image.save(path_hasil_uji_presensi+"/kelas_"+kelas+"_"+str(unix)+".jpg")
    # save firebase
    upload = storage.child("uploads/kelas/hasil/"+presensi+"/foto/kelas_"+kelas+"_"+str(unix)+".jpg").put(path_hasil_uji_presensi+"/kelas_"+kelas+"_"+str(unix)+".jpg")
    # get url gambar, simpan di url_gambar.txt
    get_url = storage.child("uploads/kelas/hasil/"+presensi+"/foto/kelas_"+kelas+"_"+str(unix)+".jpg").get_url(upload['downloadTokens'])
    data = {
        "kelas" : "kelas_"+kelas+"_"+str(unix)+".jpg",
        "url" : get_url
    }
    database.child("uploads/kelas/hasil/"+presensi).child("foto").push(data)
    


@app.route('/')
def index():
    
    return "Rest API Wajah Smart Presensi"

@app.route('/test')
def test():
    return "Testing Route"

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
    # revisi, select id mengajar, juga harus pakai id_mata_kuliah
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
                        urllib.request.urlretrieve(url, path_mahasiswa+"/"+j+"-"+file_name+".jpg") # nama file yg disimpan
                        # download gambar yg telah diupload mahasiswa
                        print("[SUCCESS] download "+file_name+".jpg berhasil")
        response.append({
            "status":"success"
        })
    else:
        print("[INFO] Data Tersedia")
        response.append({
            "status":"success"
        })
    
    return jsonify(response)

# latih data dan buat model data latih
@app.route('/latih/<kelas>/<matkul>', methods=['GET'])
def latih(kelas, matkul):
    response = []
    dir_kelas = "kelas_"+kelas+"_"+matkul
    path_kelas = os.path.join(BASEDIR, dir_kelas)
    path_hasil = os.path.join(path_kelas, "hasil")
    if not os.path.exists(path_hasil):
        response.append({
            "status":"Not found "+dir_kelas+" (Buat data dulu)"
        })
        return jsonify(response)
        # os.mkdir(path_hasil)

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
    hasil = {}
    response = []
    # direktori tambah matkul setelah kelas, utk tidak redundant
    # cek jika ada direktori yg kosong
    dir_kelas = "kelas_"+kelas+"_"+matkul
    path_kelas = os.path.join(BASEDIR, dir_kelas)
    if not os.path.exists(path_kelas):
        # print(f"[CREATE] {path_kelas}")
        response.append({
            "status":"Not found "+dir_kelas+" (Buat data dulu)"
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

    # path_data_prediksi = os.path.join(path_kelas, path_data_uji+"/"+key+".jpg")

    # download semua gambar presensi kelas
    data_kelas = database.child("uploads/kelas/"+presensi).get()
    for data in data_kelas.each():
        key = data.key()
        value = data.val()
        url = value['imageUrl']
        # download gambar semua
        print(f"[DOWNLOAD] {key}")
        urllib.request.urlretrieve(url, path_data_uji_presensi+"/"+key+".jpg")
    
    # download gambar kelas satu persatu
    # print("[DOWNLOAD] Data Uji"+path_data_prediksi)
    
    # upload_kelas = database.child("uploads/kelas/"+presensi).get()
    # url = upload_kelas.val()[key]['imageUrl']
    # file_name = upload_kelas.val()[key]['name']
    # urllib.request.urlretrieve(url, path_data_prediksi)
    # download data training jika tidak ada
    if not os.path.exists(path_hasil_latih+"/trained_knn_model.clf"):
        storage.child("uploads/kelas/latih/"+kelas+"_"+matkul+"/trained_knn_model.clf").download(path_hasil_latih+"/trained_knn_model.clf")

    for image_file in os.listdir(path_data_uji_presensi):
        full_file_path = os.path.join(path_data_uji_presensi, image_file)

        print("Prediksi gambar {}".format(image_file))

        prediksi = predict(full_file_path, model_path=path_hasil_latih+"/trained_knn_model.clf")
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

        # print hasil di terminal
        # hitung = 1
        for nama, (top, right, bottom, left), dist in prediksi:

            print("[DETECT] Wajah {} : {},{}".format(nama, left, top))
            # get hasil crop wajah
            image_uji = face_recognition.load_image_file(full_file_path)
            image_wajah = image_uji[top:bottom, left:right]
            pil_image = Image.fromarray(image_wajah)
            pil_image.save(f"{path_data_latih}/{nama}/{top}-{presensi}.jpg") # simpan hasil crop wajah di direktori
            # simpan di firebase
            req_getnim = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-nim-mahasiswa?mahasiswa='+str(nama)) # NAMA HARUS BENAR
            json_req_getnim = req_getnim.json()
            nim = json_req_getnim['master'][0]['nim']
            upload = storage.child(f"uploads/{nim}/{top}-{presensi}.jpg").put(f"{path_data_latih}/{nama}/{top}-{presensi}.jpg")
            get_url = storage.child(f"uploads/{nim}/{top}-{presensi}.jpg").get_url(upload['downloadTokens'])
            # get url foto sesuai nama
            get_url_nama = ""
            # data_mhs = database.child(f"uploads/{nim}").get()
            # for mhs in data_mhs.each():
            #     value = mhs.val()
            #     get_url_nama = value['imageUrl'] # get url terakhir dari loop
            # get crop gambar
            for gambar in os.listdir(path_data_latih+"/"+str(nama)):
                # for img_path in image_files_in_folder(os.path.join(path_data_latih, path_nama)):
                    if not os.path.exists(f"{path_data_latih}/{nama}/{nama}.jpg"): # jika sudah ada gambar yg tersimpan, maka lanjut ke folder lain
                        image = face_recognition.load_image_file(path_data_latih+"/"+nama+"/"+gambar)
                        gmbr_face_loc = face_recognition.face_locations(image)
                        if len(gmbr_face_loc) == 1:
                            # print(len(gmbr_face_loc))
                            for face_loc in gmbr_face_loc:
                                top, right, bottom, left = face_loc

                                face_image = image[top:bottom, left:right]
                                pil_images = Image.fromarray(face_image)
                                pil_images.save(f"{path_data_latih}/{nama}/{nama}.jpg") # simpan gambar wajah db
                                upload = storage.child(f"uploads/{nim}/{nama}.jpg").put(f"{path_data_latih}/{nama}/{nama}.jpg")
                                get_url_nama = storage.child(f"uploads/{nim}/{nama}.jpg").get_url(upload['downloadTokens'])
                    else:
                        upload = storage.child(f"uploads/{nim}/{nama}.jpg").put(f"{path_data_latih}/{nama}/{nama}.jpg")
                        get_url_nama = storage.child(f"uploads/{nim}/{nama}.jpg").get_url(upload['downloadTokens'])

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
                    "url_nama" : get_url_nama
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
                    "url_nama" : get_url_nama
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
            'nim' : str(x['nim'])
        }
        # cek jika sudah terisi hadir
        req_cek_presensi = requests.get('https://topapp.id/top-presence/api/v1/ruangan/find-mhs-detail-presensi?id_p='+str(presensi)+'&mhs='+str(x['nim']))
        json_req_cek_presensi = req_cek_presensi.json()
        master_req_cek_presensi = json_req_cek_presensi['master']
        if master_req_cek_presensi[0]['status'] == 'Tidak Hadir':
            print(f"[INFO] Mengisi Presensi: {x['nama']}")
            req_presensi = requests.post('https://topapp.id/top-presence/api/v1/presensi-detail/terima-presensi', data=data_presensi)
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

    return jsonify(response_hasil)

# route validasi kesalahan prediksi (hapus isi yg tidak valid)/ edit hasil.txt
@app.route('/validasi/<kelas>/<presensi>/<matkul>/<nim>', methods=['GET'])
def validasi(kelas, presensi, matkul, nim):
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
        if x['nim'] == nim:
            data.remove(x)
            print(f"[INFO] Data {x['nim']} dihapus")
        
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
    
# route Single Face

# route buat data
@app.route('/buat_auth', methods=['GET'])
def buat_auth():

    return jsonify("testing")
# route latih data
@app.route('/latih_auth', methods=['GET'])
def latih_auth():

    return jsonify("testing")
# route prediksi data
@app.route('/prediksi_auth', methods=['GET'])
def prediksi_auth():

    return jsonify("testing")
# route cek presensi detail
@app.route('/cek_auth', methods=['GET'])
def cek_auth():

    return jsonify("testing")

# Run Server
if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8000) # production topappvps
    app.run(debug=True) # development