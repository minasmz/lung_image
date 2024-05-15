import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pytorch_pretrained_vit import ViT
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Set seed values for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Custom dataset class
class COVID19Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for target_class in self.classes:
            class_dir = os.path.join(root_dir, target_class, 'images')
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, self.class_to_idx[target_class]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, target = self.samples[index]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, target


# Set the path to your dataset directory
dataset_path = '../images'

# Set the image size and batch size
img_size = 384
batch_size = 16

# Define data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Load the dataset
dataset = COVID19Dataset(root_dir=dataset_path, transform=data_transforms['train'])

# Get class labels
class_labels = dataset.classes
print(f"Class Labels: {class_labels}")

# Split the dataset into train, validation, and test sets
train_size = 200
val_size = 40
test_size = 200
remaining_size = len(dataset) - train_size - val_size - test_size

if remaining_size >= 0:
    train_dataset, val_dataset, test_dataset, _ = random_split(dataset, [train_size, val_size, test_size, remaining_size])
else:
    raise ValueError("The sum of train_size, val_size, and test_size exceeds the total size of the dataset.")


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Set the device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained ViT model
model = ViT('B_16_imagenet1k', pretrained=True, num_classes=len(class_labels))
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
patience = 8
best_f1 = 0.0
counter = 0

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print('one epoch trained')
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []

        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

        f1 = f1_score(y_true, y_pred, average='weighted')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation F1 Score: {f1:.4f}')

        if f1 > best_f1:
            best_f1 = f1
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate the model on the test set
model.eval()
y_true = []
y_pred = []
y_scores = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

        # Apply softmax to the model's output scores
        probabilities = torch.softmax(outputs, dim=1)
        y_scores.extend(probabilities.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
auc = roc_auc_score(y_true, y_scores, multi_class='ovr')

print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test F1 Score (weighted): {f1:.4f}')
print(f'Test AUC (one-vs-rest): {auc:.4f}')