import torchvision.transforms as transforms
from torchvision import datasets
import torch
from torch.utils.data import DataLoader

def data_prep(data_dir):
    data_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    data_train, data_val = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(data_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=64, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    data_train, data_val = data_prep("./data/PetImages")
    print(f"Number of training images: {len(data_train)}")
    print(f"Number of validation images: {len(data_val)}")