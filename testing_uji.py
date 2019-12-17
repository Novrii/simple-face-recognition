import face_recognition
import pickle
import os
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# gambar_dir = os.path.join(BASE_DIR, "data_latih_kelas_e")
gambar_dir = os.path.join(BASE_DIR, "img\groups")

known_face_encodings = []
known_face_names = []
# load file encodings and file names
with open("face_encodings.pickle", "rb") as face_file:
    known_face_encodings = pickle.load(face_file)

with open("face_names.pickle", "rb") as name_file:
    known_face_names = pickle.load(name_file)

# print(known_face_encodings)

# Load testing image
test_image = face_recognition.load_image_file(gambar_dir+"/bill-steve-elon.jpg")
# print(test_image)
# Find face location
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# Convert to PIL format
pil_image = Image.fromarray(test_image)

# Create a ImageDraw instance
draw = ImageDraw.Draw(pil_image)

# Loop through faces in test image
for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

  name = "Unknown Person"

  # If match
  if True in matches:
    first_match_index = matches.index(True)
    name = known_face_names[first_match_index]
  
  # Draw box
  draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))

  font = ImageFont.truetype(font="OpenSansSemibold.ttf", size=15)
  # Draw label
  text_width, text_height = draw.textsize(name)
  draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
  draw.text((left + 6, bottom - text_height - 10), name, fill=(0,0,0), font=font)

del draw

# Display image
pil_image.show()

# Save image
pil_image.save('identify.jpg')