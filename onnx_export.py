import torch
import torch.nn as nn
from torchvision import models

# load the trained model
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.load_state_dict(torch.load("signature_resnet50.pth", map_location="cpu"))
model.eval()

# 随机输入，作为导出 trace
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model, 
    dummy_input, 
    "signature_resnet50.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=12
)

print("✅ output signature_resnet50.onnx")
