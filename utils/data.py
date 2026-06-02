import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_dataloaders_MNIST(ranges):
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)

    def subset(ds, lo, hi):
        mask = (ds.targets >= lo) & (ds.targets <= hi)
        return Subset(ds, torch.where(mask)[0])

    train_loaders = [DataLoader(subset(mnist_train, lo, hi), batch_size=64, shuffle=True) for lo, hi in ranges]
    test_loaders = [DataLoader(subset(mnist_test, lo, hi), batch_size=64, shuffle=False) for lo, hi in ranges]
    return train_loaders, test_loaders

def get_dataloaders_CIFAR100(
    num_tasks=10,
    classes_per_task=10,
    batch_size=64,
    data_dir="./data"
):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761)
        )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761)
        )
    ])

    cifar_train = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train
    )

    cifar_test = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test
    )

    def subset_by_classes(dataset, class_ids):
        targets = torch.tensor(dataset.targets)

        mask = torch.zeros_like(targets, dtype=torch.bool)

        for c in class_ids:
            mask |= (targets == c)

        indices = torch.where(mask)[0]

        return Subset(dataset, indices)

    train_loaders = []
    test_loaders = []

    for task_id in range(num_tasks):

        start_class = task_id * classes_per_task
        end_class = start_class + classes_per_task

        class_ids = list(range(start_class, end_class))

        train_subset = subset_by_classes(cifar_train, class_ids)
        test_subset = subset_by_classes(cifar_test, class_ids)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )

        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders
