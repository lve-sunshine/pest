# 基于Pytorch实现的昆虫识别小程序（微信小程序  + Flask）
本项目是基于Pytorch，根据`Do`数据集训练得到的40分类昆虫模型，基于`flask`框架开放请求接口与响应的昆虫识别小程序。
## 1. 效果展示
![image](https://github.com/lve-sunshine/pest/assets/99074010/33160363-7538-4203-bd2a-1ff452bc3774)

![image](https://github.com/lve-sunshine/pest/assets/99074010/dd035e85-a986-49d6-a4ba-708c9c5bfe54)

![image](https://github.com/lve-sunshine/pest/assets/99074010/a65a3a6c-1d45-458e-a238-517d47b29f1b)

## 2. 后端
简单的使用`flask`框架，加载已训练好的昆虫识别model，开放api接口接受识别请求并返回识别结果。
```python
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
    app.run(debug=True, host='0.0.0.0')

```


