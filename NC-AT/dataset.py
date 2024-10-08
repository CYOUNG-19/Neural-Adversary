from torchvision import transforms
import torchvision
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        if split == 'train':
            self.data_dir = os.path.join(root_dir, 'train')
            self.image_paths, self.labels = self._load_train()
        elif split == 'val':
            self.data_dir = os.path.join(root_dir, 'val')
            self.image_paths, self.labels = self._load_val()
        elif split == 'test':
            self.data_dir = os.path.join(root_dir, 'test')
            self.image_paths = self._load_test()
            self.labels = None

    def _load_train(self):
        image_paths = []
        labels = []
        classes = sorted(os.listdir(self.data_dir))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        for cls_name in classes:
            cls_folder = os.path.join(self.data_dir, cls_name, 'images')
            image_files = os.listdir(cls_folder)
            image_paths.extend([os.path.join(cls_folder, img) for img in image_files])
            labels.extend([class_to_idx[cls_name]] * len(image_files))
        return image_paths, labels

    def _load_val(self):
        image_paths = []
        labels = []
        val_images_dir = os.path.join(self.data_dir, 'images')
        val_annotations_file = os.path.join(self.data_dir, 'val_annotations.txt')
        with open(val_annotations_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                image_paths.append(os.path.join(val_images_dir, parts[0]))
                labels.append(parts[1])
        classes = sorted(list(set(labels)))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        labels = [class_to_idx[cls] for cls in labels]
        return image_paths, labels

    def _load_test(self):
        image_paths = []
        test_images_dir = os.path.join(self.data_dir, 'images')
        image_files = os.listdir(test_images_dir)
        image_paths.extend([os.path.join(test_images_dir, img) for img in image_files])
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image


def return_dataloader(args):

    if args.dataset == 'cifar10':

        cifar_transform = {
            'train': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
            ]),
            }

        # CIFAR-10 train and test datasets
        cifar10_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform['train'])
        cifar10_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform['test'])

        # CIFAR-10 train and test dataloaders
        train_loader = DataLoader(cifar10_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(cifar10_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
    elif args.dataset == 'cifar100':

        cifar_transform = {
            'train': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
            ]),
            }

        # CIFAR-100 train and test datasets
        cifar100_train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=cifar_transform['train'])
        cifar100_test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=cifar_transform['test'])

        # CIFAR-100 train and test dataloaders
        train_loader = DataLoader(cifar100_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(cifar100_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    elif args.dataset == 'imagenet':
        imagenet_transform = {
            'train': transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ]),
        }

        # ImageNet train and validation datasets
        train_dataset = TinyImageNetDataset(root_dir='/home/cy/uncat/tiny-imagenet-200', split='train', transform=imagenet_transform['train'])

        # ImageNet train and validation dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)


    else:
        raise ValueError(f"Unsupported model name: {args.dataset}")

    if args.dataset == 'cifar10':
        return train_loader, test_loader
    elif args.dataset == 'cifar100':
        return train_loader, test_loader
    elif args.dataset == 'imagenet':
        return train_loader