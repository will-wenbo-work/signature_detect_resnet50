import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# 1. 创建 ONNX Runtime session
sess = onnxruntime.InferenceSession("signature_resnet.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name

# 2. 准备一张图片用于推理
# 假设你已经有了一张裁剪好的图片 'cropped_image.png'
image = Image.open('cropped_image.png').convert('RGB')

# 3. 对图片进行预处理，使其符合模型输入要求
# 注意，这里的预处理要和训练时保持一致
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

input_data = transform(image).unsqueeze(0).numpy() # 转换为 numpy 数组

# 4. 执行推理
outputs = sess.run(None, {input_name: input_data})
output = outputs[0]

# 5. 处理推理结果
# 这里我们假设模型输出的是概率分布
probabilities = np.exp(output) / np.sum(np.exp(output))
class_labels = ['non_signature', 'signature']
predicted_class_index = np.argmax(probabilities)
predicted_class = class_labels[predicted_class_index]

print(f"Predicted class: {predicted_class}")
print(f"Probabilities: {probabilities}")