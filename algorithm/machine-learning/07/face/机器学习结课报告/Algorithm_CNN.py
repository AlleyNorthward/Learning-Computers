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
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm.auto import tqdm
from manim import*

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False  

class ORLDataset(Dataset):
    def __init__(
            self,
            root,
            train = True,
            k = 5,
            transform = None,
            n = 40
    ):

        self.transform = transform
        self.data, self.labels = self.init_data_labels(root, train, k, n)
            
    def init_data_labels(self, root, train, k, n):
        data = []
        labels = []

        for i in range(n):
            person = f"s{i+1}"
            imgs = [f"{j}.BMP" for j in range(1, 11)]
            if train:
                selected = imgs[:k]
            else:
                selected = imgs[k:]
            
            for img_name in selected:
                img_path = os.path.join(root, person, img_name)
                data.append(img_path)
                labels.append(i)
            
        return data, labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
    
    def print_data(self, num = 5):
        infos = "查看内部存储的data信息:"
        print(infos)
        pprint(self.data[:num])
        print(len(self))

    def print_label(self, num = 5):
        infos = "查看内部储存的label信息:"
        print(infos)
        pprint(self.labels[:num])
        print(len(self.labels))

    def analysis_image_label(self, idx):
        image, label = self[idx]
        pprint(image)
        print(image.shape)
        print(f"这是第{label}人.")
    
transform = transforms.Compose([
    transforms.Resize((112, 92)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

root = str(Path(__file__).resolve().parent / "dataset" / "bmp")

train_dataset = ORLDataset(root, k = 6, transform=transform, n = 45)
test_dataset = ORLDataset(root, train=False, k = 6, transform=transform, n = 45)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class FaceCNN(nn.Module):
    def __init__(self, num_classes=40):
        super(FaceCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # 卷积核大小3x3 卷积
        self.pool = nn.MaxPool2d(2,2) # 缩小比例 池化
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) 
        self.fc1 = nn.Linear(64*28*23, 256)# 全连接层, 线性层 最后根据这个,可以得出
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 112x92 -> 56x46
        x = self.pool(F.relu(self.conv2(x)))  # 56x46 -> 28x23
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # 抽象为结果得分
        return x
    
class Train:
    def __init__(
            self,
            num_epochs = 20
    ):
        self.init_infos = self._prepare_all()
        self.model = self.init_infos[1]
        self.train_losses = self.train(self.init_infos, num_epochs)
        self.accuracy = self.train_accuracy(self.init_infos)
        
    def _prepare_all(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = FaceCNN(num_classes=45).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr = 0.001)
        
        init_infos = [device, model, criterion, optimizer]
        return init_infos
    
    def train(self, init_infos, num_epochs = 20):
        device, model, criterion, optimizer = init_infos

        train_losses = []
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            loader = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", unit="batch", ncols=100)
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                loader.set_postfix(loss=loss.item())
            train_losses.append(f"{running_loss/len(train_loader):.4f}")
        return train_losses
    
    def get_train_plot(self):
        train_losses_float = [float(x) for x in self.train_losses]
        plt.figure(figsize=(8,5))
        plt.plot(train_losses_float, marker='o', linestyle='-', color='blue')
        plt.title("学习损失曲线")
        plt.xlabel("轮次")
        plt.ylabel("平均损失")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def train_accuracy(self, init_infos):
        device, model, _, _ = init_infos
        model.eval()
        correct = 0
        total = 0
        accuracy = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                accuracy.append(f"{100*correct / total:.2f}")

        return accuracy

    def get_train_accuracy_plot(self):
        accuracy_float = [float(x) for x in self.accuracy]

        plt.figure(figsize = (8, 5))
        plt.plot(accuracy_float, marker='o', color = 'red')
        plt.xlabel("测试数据集")
        plt.ylabel("准确性 (%)")
        plt.title("测试集的准确性")
        plt.grid(True)
        plt.ylim(50, 100)
        plt.show()

class RealTimeDetection:
    def __init__(self, infos, using_real_name = False):
        real_names = ["wly", 'wql', 'wwd', 'wxz', 'ykj']
        person_names = [f"Person {i+1}" for i in range(40)]
        if using_real_name:
            person_names.extend(real_names)
        
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.start(person_names, cap, face_cascade,infos)

    def start(self, person_names, cap, face_cascade, infos):
        device, model, _, _ = infos
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
                    face_img = transform(face_img).unsqueeze(0).to(device)
                    outputs = model(face_img)
                    _, predicted = torch.max(outputs.data, 1)
                    name = person_names[predicted.item()]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                cv2.imshow("Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()


#分析查看接口
# train_dataset.print_data()
# train_dataset.print_label()

# train_dataset.analysis_image_label(0)
# train_dataset.analysis_image_label(0)

for image, label in train_loader:
    print(image.shape)
    print(f"一共有{len(label)}, 分别是:{label}.")
    print(f"一共有{len(train_loader)}个batch.")
    break

# train = Train()
# print(train.train_losses)
# train.get_train_plot()
# train.get_train_accuracy_plot()
# real = RealTimeDetection(train.init_infos, using_real_name=True)


# class LossAccuracyAnimation(Scene):
#     def construct(self):
#         self.camera.background_color = '#cee'
#         axes = Axes(
#             x_range=[0, len(train.train_losses), 1],
#             y_range=[0, 4, 0.1],
#             x_length=8,
#             y_length=6,
#             axis_config={"color": BLUE},
#             x_axis_config={
#                 "numbers_to_include":np.arange(0,len(train.train_losses), 2)
#             },
#             y_axis_config={
#                 "numbers_to_include": np.arange(0, 4, 0.5)
#             },
#             tips=False
#         ).to_edge(DOWN)

#         labels = axes.get_axis_labels(x_label="Epoch", y_label="Loss")
#         axes.set_color(BLACK)
#         labels.set_color(BLUE_B)

#         loss_points = [axes.coords_to_point(i, float(loss)) for i, loss in enumerate(train.train_losses)]
#         loss_dots = [Dot(p, color=RED) for p in loss_points]

#         loss_line = VMobject(color=RED)
#         loss_line.set_points_as_corners([loss_points[0]])

#         title = Text("训练损失变化", font_size=36).to_edge(UP)
#         title.set_color(TEAL)

#         self.play(Write(title))
#         self.play(Create(axes), Write(labels))

#         for i in range(1, len(loss_points)):
#             new_segment = Line(loss_points[i - 1], loss_points[i], color=RED)
#             self.play(FadeIn(new_segment), FadeIn(loss_dots[i]), run_time=0.3, subcaption_offset=0.1)
#             loss_line.add_points_as_corners([loss_points[i]])

#         self.wait(5)

#         # =============================
#         # =============================

#         self.clear()
#         axes2 = Axes(
#             x_range=[0, len(train.accuracy), 1],
#             y_range=[40, 100, 10],
#             x_length=8,
#             y_length=6,
#             axis_config={"color": GREEN},
#             x_axis_config={
#                 "numbers_to_include": np.arange(0, len(train.accuracy), 2)
#             },
#             y_axis_config={
#                 "numbers_to_include": np.arange(40, 100, 10)
#             },
#             tips=False
#         ).to_edge(DOWN)

#         labels2 = axes2.get_axis_labels(x_label="Batch", y_label="Accuracy")
#         labels2.set_color(BLUE_B)
#         axes.set_color(BLACK)
#         acc_points = [axes2.coords_to_point(i, float(acc)) for i, acc in enumerate(train.accuracy)]
#         acc_dots = [Dot(p, color=YELLOW) for p in acc_points]

#         acc_line = VMobject(color=YELLOW)
#         acc_line.set_points_as_corners([acc_points[0]])

#         title2 = Text("测试准确率变化", font_size=36).to_edge(UP)
#         title2.set_color(TEAL)

#         self.play(Write(title2))
#         self.play(Create(axes2), Write(labels2))

#         for i in range(1, len(acc_points)):
#             new_segment = Line(acc_points[i - 1], acc_points[i], color=YELLOW)
#             self.play(Create(new_segment), FadeIn(acc_dots[i]), run_time=0.3)
#             acc_line.add_points_as_corners([acc_points[i]])

#         self.wait(3)
