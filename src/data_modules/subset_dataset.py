import torch


class SubsetDataset(torch.utils.data.Dataset):
    """
    Applies transformation for the dataset subset.
    This was necessary as the Flowers102 dataset was unbalanced,
    so we needed to redistribute it and apply the transformation after that.
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)
