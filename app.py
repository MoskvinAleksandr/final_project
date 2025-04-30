from flask import Flask, render_template, request
import torch
from PIL import Image
import torchvision.transforms as transforms
from io import BytesIO
import base64
import cv2
import numpy as np
from PIL import UnidentifiedImageError
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
app = Flask(__name__)

def crop_face(pil_img):
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_GRAY2BGR)

    faces = face_cascade.detectMultiScale(img_cv, scaleFactor=1.1, minNeighbors=5)

    (x, y, w, h) = faces[0]

    face_img = img_cv[y:y+h, x:x+w]

    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY))
    return face_pil
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(128 * 6 * 6, 512)
        self.fc2 = torch.nn.Linear(512, 7)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 128 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()
model.load_state_dict(torch.load('emotion_model.pth', map_location=torch.device('cpu')))
model.eval()
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#@app.route('/detailed', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_file = request.files['image']
     #   print("Получили файл:", image_file.filename)
      #  print("Контент-тайп:", image_file.content_type)

        #try:
        image = Image.open(image_file.stream).convert('RGB')
      #  except Exception as e:
      #      print("Ошибка при открытии изображения:", e)
          #  return render_template('index.html', emotion="Не удалось открыть изображение", image_data=None)

        open_cv_image = np.array(image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        img_io_original = BytesIO()
        image.save(img_io_original, 'JPEG')
        img_io_original.seek(0)
        img_data_original = base64.b64encode(img_io_original.getvalue()).decode('utf-8')

        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            return render_template('index.html', emotion="Лицо не найдено", image_data=None)

        (x, y, w, h) = faces[0]

        img_with_box = open_cv_image.copy()
        cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img_with_box_pil = Image.fromarray(cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB))
        img_io_box = BytesIO()
        img_with_box_pil.save(img_io_box, 'JPEG')
        img_io_box.seek(0)
        img_data_box = base64.b64encode(img_io_box.getvalue()).decode('utf-8')

        face_img = gray[y:y+h, x:x+w]
        face_pil = Image.fromarray(face_img).convert('L')

        img_io_face = BytesIO()
        face_pil.save(img_io_face, 'JPEG')
        img_io_face.seek(0)
        img_data_face = base64.b64encode(img_io_face.getvalue()).decode('utf-8')

        input_tensor = transform(face_pil).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, 1).item()
        emotion = emotion_labels[predicted_class]
        probabilities = torch.softmax(output, dim=1).squeeze().tolist()
        emotion_probabilities = sorted(zip(emotion_labels, probabilities), key=lambda x: x[1], reverse=True)
        return render_template(
            'index.html',
            emotion=emotion,
            img_original=img_data_original,
            img_with_box=img_data_box,
            img_face=img_data_face,
            emotion_probabilities=emotion_probabilities
        )

    return render_template('index.html', emotion=None)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

