import face_recognition
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# gambar_dir = os.path.join(BASE_DIR, "data_latih_kelas_e")
gambar_dir = os.path.join(BASE_DIR, "img\known")

known_face_encodings = []
known_face_names = []

for root, dirs, files in os.walk(gambar_dir):
    for file in files:
        if file.endswith("jpg"): # only jpg file
            # print(os.path.join(root, file))
            path = os.path.join(root, file)
            base_name = os.path.basename(path)
            name = base_name.strip('.jpg')
            label = name.replace(" ", "")
            # split base_name to make an array of string 
            # use cv2 to resize image
            
            # print(label)
            print(f"[INFO] Load {path}")
            load_image = face_recognition.load_image_file(path)
            location_face = face_recognition.face_locations(load_image, model="cnn")
            face_encoding = face_recognition.face_encodings(load_image, location_face)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)

# print(known_face_encodings)
# write pickle file face encodings and file name
with open("face_encodings.pickle", "wb") as f:
    pickle.dump(known_face_encodings, f)
    print("[SUCCESS] Saved file face_encodings.pickle")

with open("face_names.pickle", "wb") as name_file:
    pickle.dump(known_face_names, name_file)
    print("[SUCCESS] Saved file face_names.pickle")

