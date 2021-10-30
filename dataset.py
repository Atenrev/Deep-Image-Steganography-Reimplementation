import os
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTENSIONS = ["jpeg", "jpg", "png"]


class CustomDataset(Dataset):
    """
    Image dataset.
    """

    def __init__(self, root_dir: str, transform=None) -> None:
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.src_files = list()

        for root, d_names, f_names in os.walk(root_dir):
            for f in f_names:
                if f.split('.')[-1].lower() in IMG_EXTENSIONS:
                    self.src_files.append(os.path.join(root, f))

        self.transform = transform

    def __len__(self) -> int:
        return 1000
        return len(self.src_files)

    def __getitem__(self, idx: int):
        img_name = self.src_files[idx]
        image = Image.open(str(img_name)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image
