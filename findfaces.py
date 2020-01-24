import face_recognition
import os
import cv2
import json

image = face_recognition.load_image_file('./img/groups/team2.jpg')
face_locations = face_recognition.face_locations(image)

# fungsi rescale image 
def scale_down(scale, img):
    width = int(img.shape[1] * scale/100)
    height = int(img.shape[0] * scale/100)
    dim = (width, height)
    new_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return new_image
# fungsi cek_size
def cek_size(size, img):
    if size > 100000 and size < 200000:
        scale_percent = 70
        rescale = scale_down(scale_percent, img)
        cv2.imwrite("test.jpg", rescale)
        new_size = os.path.getsize("test.jpg")
        print(f"New Size: {new_size/1024} (100)")
    elif size > 200000 and size < 300000:
        scale_percent = 60
        rescale = scale_down(scale_percent, img)
        cv2.imwrite("test.jpg", rescale)
        new_size = os.path.getsize("test.jpg")
        print(f"New Size: {new_size/1024} (200)")
    elif size > 300000 and size < 400000:
        scale_percent = 50
        rescale = scale_down(scale_percent, img)
        cv2.imwrite("test.jpg", rescale)
        new_size = os.path.getsize("test.jpg")
        print(f"New Size: {new_size/1024} (300)")
    elif size > 400000 and size < 500000:
        scale_percent = 40
        rescale = scale_down(scale_percent, img)
        cv2.imwrite("test.jpg", rescale)
        new_size = os.path.getsize("test.jpg")
        print(f"New Size: {new_size/1024} (400)")
    elif size > 500000:
        scale_percent = 30
        rescale = scale_down(scale_percent, img)
        cv2.imwrite("test.jpg", rescale)
        new_size = os.path.getsize("test.jpg")
        print(f"New Size: {new_size/1024} (500)")

# Array of coords of each face
# print(face_locations)

# print(f'There are {len(face_locations)} people in this image')

# path = './data_latih_kelas_e/MUHAMMAD HAFIZD ANSHARI/-Lvx1f7zABzldy7K4iqy-saya2.jpg'
# # testing size
# size = os.path.getsize(path)
# print(f"Size: {size/1024}")
# # resize
# gray_img = cv2.imread(path)

# cek_size(size, gray_img)

data = {}

with open('hasil.txt') as json_file:
    data = json.load(json_file)

data[len(data)+1]={
    "nama":"testing2"
}

with open('hasil.txt', 'w') as json_file:
    json.dump(data, json_file)

print(len(data))