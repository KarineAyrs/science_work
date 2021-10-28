from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class MNIST:
    def __init__(self, root, batch_size):
        self.batch_size = batch_size
        self._train_dataset = datasets.MNIST(root=root, train=True, transform=transforms.ToTensor(),
                                             download=True)
        self._train_loader = DataLoader(dataset=self._train_dataset, batch_size=batch_size, shuffle=True)

        self._test_dataset = datasets.MNIST(root=root, train=False, transform=transforms.ToTensor(),
                                            download=True)
        self._test_loader = DataLoader(dataset=self._test_dataset, batch_size=batch_size, shuffle=True)

    def train_loader(self):
        return self._train_loader

    def test_loader(self):
        return self._test_loader
