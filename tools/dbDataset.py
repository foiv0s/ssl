from typing import Union
from torch.utils.data import Dataset
from pathlib import Path
from tools.database import ImageDatabase, LabelDatabase


class dbDataset(Dataset):
    def __init__(self, path: Union[str, Path], transform=None, target_transform=None):
        if not isinstance(path, Path):
            path = Path(path)

        # images = path / f"Images.mdb"
        # labels = path / f"Labels.mdb"
        images = path / f"imgs"
        labels = path / f"lbls"

        self.images = ImageDatabase(path=images)
        self.labels = LabelDatabase(path=labels)
        self.keys = self._keys()
        self.transform = transform
        self.target_transform = target_transform

    def _keys(self):
        # We assume that the keys are the same for the images and the labels.
        # Feel free to do something else if you fancy it.
        keys = sorted(set(self.images.keys).intersection(self.labels.keys))
        return keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        key = self.keys[item]

        '''
        data = {
            "image": self.images[key],
            "label": self.labels[key]
        }
        if self.transform:
            data = self.transform(data)
        return data
        '''

        target = self.labels[key]
        sample = self.images[key].convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, self.keys.index(key)
