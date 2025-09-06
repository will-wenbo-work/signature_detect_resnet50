import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import argparse
import os

class SignatureResNetTrainer:
    def __init__(self, data_dir, output_dir, num_epochs=10, lr=1e-3, batch_size=16):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataloaders, self.image_datasets = self.get_data_loaders()
        self.class_names = self.image_datasets['train'].classes
        print("Classes:", self.class_names)
        self.model = self.build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=self.lr)

    def get_data_loaders(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
        }
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'val']}
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=self.batch_size,
                                     shuffle=True, num_workers=4)
                       for x in ['train', 'val']}
        return dataloaders, image_datasets

    def build_model(self):
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False  # 冻结卷积层
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(self.class_names))  # 适配分类数
        return model.to(self.device)

    def train(self):
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                running_loss, running_corrects = 0.0, 0
                for inputs, labels in self.dataloaders[phase]:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                epoch_loss = running_loss / len(self.image_datasets[phase])
                epoch_acc = running_corrects.double() / len(self.image_datasets[phase])
                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    def export(self):
        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "signature_resnet50.pth"))
        print("✅ 已保存 PyTorch 模型权重")
        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        torch.onnx.export(
            self.model,
            dummy_input,
            os.path.join(self.output_dir, "signature_resnet50.onnx"),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=12
        )
        print("✅ export ONNX model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset folder (with train/ and val/)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Where to save models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    trainer = SignatureResNetTrainer(args.data_dir, args.output_dir, args.epochs, args.lr)
    trainer.train()
    trainer.export()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset folder (with train/ and val/)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Where to save models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_and_export(args.data_dir, args.output_dir, args.epochs, args.lr)
