# coding:utf-8
import os
from pathlib import Path
from PIL import Image
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

# ------------------------------
# 数据集
# ------------------------------
class ORLDataset(Dataset):
    def __init__(self, root, train=True, k=5, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        for i in range(40):
            person = f"s{i+1}"
            imgs = [f"{j}.BMP" for j in range(1, 11)]
            if train:
                selected = imgs[:k]
            else:
                selected = imgs[k:]
            for img_name in selected:
                img_path = os.path.join(root, person, img_name)
                self.data.append(img_path)
                self.labels.append(i)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('L')  # 灰度
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

transform = transforms.Compose([
    transforms.Resize((112, 92)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

orl_path = str(Path(__file__).resolve().parent / "dataset" / "bmp")
train_dataset = ORLDataset(orl_path, train=True, k=5, transform=transform)
test_dataset = ORLDataset(orl_path, train=False, k=5, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ------------------------------
# CNN 模型
# ------------------------------
class FaceCNN(nn.Module):
    def __init__(self, num_classes=40):
        super(FaceCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*28*23, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 112x92 -> 56x46
        x = self.pool(F.relu(self.conv2(x)))  # 56x46 -> 28x23
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ------------------------------
# 训练与测试
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FaceCNN(num_classes=40).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# 测试准确率
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"测试集准确率: {100*correct/total:.2f}%")

# ------------------------------
# 实时摄像头刷脸识别
# ------------------------------
# ORL 标签映射为人名或编号
person_names = [f"Person {i+1}" for i in range(40)]

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model.eval()
with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = Image.fromarray(face_img).convert('L')
            face_img = transform(face_img).unsqueeze(0).to(device)  # 1x1x112x92
            outputs = model(face_img)
            _, predicted = torch.max(outputs.data, 1)
            name = person_names[predicted.item()]
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
