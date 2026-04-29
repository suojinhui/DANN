import torch.utils.data as data
from PIL import Image
import os


class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            data = data.strip()
            if not data:
                continue

            parts = data.rsplit(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid data list entry: '{data}'")

            img_path, img_label = parts
            self.img_paths.append(img_path)
            self.img_labels.append(int(img_label))

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels

    def __len__(self):
        return self.n_data

if __name__ == '__main__':
    pass
