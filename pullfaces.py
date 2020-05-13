from PIL import Image, ImageDraw
import face_recognition

image = face_recognition.load_image_file('test/4.jpg')
face_locations = face_recognition.face_locations(image)

# Convert to PIL format
pil_image = Image.fromarray(image)

# Create a ImageDraw instance
draw = ImageDraw.Draw(pil_image)

print(face_locations)

for face_location in face_locations:
    top, right, bottom, left = face_location

    # face_image = image[top:bottom, left:right]
    # pil_image = Image.fromarray(face_image)
    # pil_image.show()
    # Draw box
    draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))
    # pil_image.save(f'knn/{top}.jpg')

del draw

pil_image.show()
