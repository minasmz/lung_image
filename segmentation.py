import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
from sklearn.metrics import accuracy_score, f1_score, jaccard_score

# Set seed values for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Custom dataset class
class LungSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))

        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            image_dir = os.path.join(class_dir, 'images')
            mask_dir = os.path.join(class_dir, 'masks')

            for img_name in os.listdir(image_dir):
                image_path = os.path.join(image_dir, img_name)
                mask_path = os.path.join(mask_dir, img_name)
                self.samples.append((image_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# Set the path to your dataset directory
dataset_path = '../images'

# Set the image size and batch size
img_size = 256
batch_size =16

# Define data transforms
data_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load the dataset
dataset = LungSegmentationDataset(root_dir=dataset_path, transform=data_transforms)

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

# Load the U-Net model
model = smp.Unet('resnet34', encoder_weights='imagenet', activation='sigmoid', in_channels=1, classes=1)
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)


# Train the model
num_epochs = 50
patience = 5
best_iou = 0.0
counter = 0

for epoch in range(num_epochs):
    model.train()
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

    print('one epoch trained')
    model.eval()
    with torch.no_grad():
        ious = []

        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()

            iou = jaccard_score(masks.flatten().cpu().numpy(), preds.flatten().cpu().numpy(), average='micro')
            ious.append(iou)

        mean_iou = sum(ious) / len(ious)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Mean IoU: {mean_iou:.4f}')

        scheduler.step(mean_iou)

        if mean_iou > best_iou:
            best_iou = mean_iou
            counter = 0
            torch.save(model.state_dict(), 'best_model_seg.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

# Load the best model
model.load_state_dict(torch.load('best_model_seg.pth'))

# Evaluate the model on the test set
model.eval()
ious = []

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        preds = (outputs > 0.5).float()

        iou = jaccard_score(masks.flatten().cpu().numpy(), preds.flatten().cpu().numpy(), average='micro')
        ious.append(iou)

mean_iou = sum(ious) / len(ious)
print(f'Test Mean IoU: {mean_iou:.4f}')