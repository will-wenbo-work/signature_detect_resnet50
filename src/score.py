import json
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import base64

# init ONNX Runtime
def init():
    global session
    session = ort.InferenceSession("signature_resnet50.onnx", providers=["CPUExecutionProvider"])

# 处理请求
def run(raw_data):
    try:
        # input base64 code image
        data = json.loads(raw_data)
        img_bytes = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image = image.resize((224, 224))

        # numpy
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = (img_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
        img_array = np.expand_dims(img_array, axis=0)

        # ONNX inference
        outputs = session.run(None, {"input": img_array})
        probs = np.squeeze(outputs[0])
        pred_class = int(np.argmax(probs))

        return json.dumps({
            "prediction": "signed" if pred_class == 1 else "unsigned",
            "confidence": float(np.max(probs))
        })

    except Exception as e:
        return json.dumps({"error": str(e)})
