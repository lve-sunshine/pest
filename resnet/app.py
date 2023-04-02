import myModel
from PIL import Image
from flask import Flask, request, jsonify
from io import BytesIO
import base64
import torch
import json

model = myModel.load_model()
# 开启评估模式
model.eval()
app = Flask(__name__)


@app.route('/inference', methods=['POST'])
def insect_recognition():
    # Parse the input image data from the request
    image_data = request.json['image_data']

    # Convert the base64-encoded image data to a PIL image
    image = Image.open(BytesIO(base64.b64decode(image_data)))

    # Apply the image transformation
    image_tensor = myModel.tf(image)

    # Add batch dimension to the input image
    image_tensor = image_tensor.unsqueeze(0)

    # Make predictions
    with torch.no_grad():
        outputs = model(image_tensor)

    # Apply softmax to the outputs to get class probabilities
    probs = torch.nn.functional.softmax(outputs, dim=1)
    value, index = torch.max(probs[0], dim=-1)
    accuracy = round(float(value.item()), 4) * 100
    # # Get the predicted class and probability
    # _, predicted = torch.max(probs, dim=1)
    # predicted_class = predicted.item()
    # probability = probs[0][predicted_class].item()

    # Return the predicted class and probability in a JSON response
    response = {'class': index.item(), 'probability': '%.2f' % accuracy}
    return jsonify(response)


@app.route('/pest', methods=['POST'])
def subJsonByNo():
    no = request.json['no']
    with open(f'./PestInformation/data/{no}.json', 'r') as f:
        return json.load(f)


@app.route('/test', methods=['get'])
def test():
    return jsonify({'info': 'success'})


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
