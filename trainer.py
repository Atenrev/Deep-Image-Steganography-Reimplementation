import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from dataset import CustomDataset
from model import AutoSteganographer


def _parse_args() -> object:
    parser = argparse.ArgumentParser(
        description='Steganography trainer parser')

    parser.add_argument('--train_dataset', type=str, default='./tiny_imagenet/train',
                        help='location of the dataset')
    parser.add_argument('--val_dataset', type=str, default='./tiny_imagenet/val',
                        help='location of the dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='training batch size')
    parser.add_argument('--image_size', type=int, default=64,
                        help='image size')
    parser.add_argument('--val_perc', type=float, default=0.1,
                        help='validation data percentage')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='adam beta')
    parser.add_argument('--save_dir', type=str, default='./model',
                        help='save model directory')
    parser.add_argument('--load_model', type=str, default=None,
                        help='model to load and resume training')

    args = parser.parse_args()
    return args


class Trainer:
    model: AutoSteganographer
    train_loader: DataLoader
    val_loader: DataLoader
    save_path: str
    device: torch.device
    loss_fn: object
    optimizer: object
    lr: float
    epochs: int

    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 save_path: str = "./model",
                 learning_rate: float = 0.001,
                 num_epochs: int = 10,
                 adam_beta: float = 0.5
                 ) -> None:
        # torch.manual_seed(42)

        self.save_path = save_path
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = learning_rate
        self.loss_fn = torch.nn.MSELoss()
        self.num_epochs = num_epochs

        self.optimizer_merger = torch.optim.Adam(
            self.model.merger.parameters(), lr=self.lr, betas=(adam_beta, 0.999))
        self.optimizer_decoder = torch.optim.Adam(
            self.model.revealer.parameters(), lr=self.lr, betas=(adam_beta, 0.999))

        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f'Selected device: {self.device}')

        self.model.to(self.device)

    def train_epoch(self) -> float:
        self.model.merger.train()
        self.model.revealer.train()
        train_loss = []

        for image_batch in tqdm(self.train_loader):
            batch_size = image_batch.shape[0] // 2
            original_image_batch = image_batch[0:batch_size].to(self.device)
            hidden_image_batch = image_batch[batch_size:batch_size*2].to(
                self.device)

            encoded_image, decoded_image = self.model(
                original_image_batch, hidden_image_batch)

            merger_loss = self.loss_fn(encoded_image, original_image_batch)
            reveal_loss = self.loss_fn(
                decoded_image, hidden_image_batch) * 0.75

            loss = reveal_loss + merger_loss

            self.optimizer_merger.zero_grad()
            self.optimizer_decoder.zero_grad()

            loss.backward()

            self.optimizer_merger.step()
            self.optimizer_decoder.step()

            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)

    def test_epoch(self) -> float:
        self.model.eval()

        with torch.no_grad():
            i = 0
            total_loss = 0.0

            for image_batch in self.val_loader:
                batch_size = image_batch.shape[0] // 2
                original_image_batch = image_batch[0:batch_size].to(
                    self.device)
                hidden_image_batch = image_batch[batch_size:batch_size*2].to(
                    self.device)

                encoded_image, decoded_image = self.model(
                    original_image_batch, hidden_image_batch)

                merger_loss = self.loss_fn(encoded_image, original_image_batch)
                reveal_loss = self.loss_fn(
                    decoded_image, hidden_image_batch) * 0.75

                total_loss += (reveal_loss + merger_loss).data

                i += 1

            val_loss = total_loss / i

        return val_loss

    def plot_ae_outputs(self, n: int = 5) -> None:
        plt.figure(figsize=(10, 5))
        i = 0

        for image_batch in self.val_loader:
            batch_size = image_batch.shape[0] // 2
            original_image_batch = image_batch[0:batch_size].to(self.device)
            hidden_image_batch = image_batch[batch_size:batch_size*2].to(
                self.device)

            ax = plt.subplot(4, n, i + 1)
            plt.imshow(np.moveaxis(original_image_batch[0].cpu().squeeze().numpy(), 0, -1))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            if i == n//2:
                ax.set_title('Original images')

            ax = plt.subplot(4, n, i + 1 + n)
            plt.imshow(np.moveaxis(hidden_image_batch[0].cpu().squeeze().numpy(), 0, -1))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            if i == n//2:
                ax.set_title('Hidden images')

            self.model.eval()

            with torch.no_grad():
                enc_img, rec_img = self.model(
                    original_image_batch, hidden_image_batch)

            ax = plt.subplot(4, n, i + 1 + n * 2)
            plt.imshow(np.moveaxis(enc_img[0].cpu().squeeze().numpy(), 0, -1))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            if i == n//2:
                ax.set_title('Merged images')

            ax = plt.subplot(4, n, i + 1 + n * 3)
            plt.imshow(np.moveaxis(rec_img[0].cpu().squeeze().numpy(), 0, -1))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            if i == n//2:
                ax.set_title('Reconstructed images')

            i += 1

            if i >= n:
                break

        plt.show()

    def train(self) -> None:
        val_loss = 0

        for epoch in range(self.num_epochs):
            print(
                '\n\n -------- RUNNING EPOCH {}/{} --------\n'.format(epoch + 1, self.num_epochs))
            train_loss = self.train_epoch()
            val_loss = self.test_epoch()
            print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch +
                                                                          1, self.num_epochs, train_loss, val_loss))

        torch.save(self.model.state_dict(), os.path.join(
            self.save_path, "autosteganographer.pth"))

        self.plot_ae_outputs(n=5)


def main(args) -> None:
    # Prepare the data
    train_transform = transforms.Compose([
        transforms.Resize([args.image_size, args.image_size]),
        transforms.ToTensor(),
    ])
    train_data = CustomDataset(
        args.train_dataset,
        transform=train_transform
    )
    val_data = CustomDataset(
        args.val_dataset,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=True)

    # Define the model
    model = AutoSteganographer()

    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model))

    # Trainer
    trainer = Trainer(model, train_loader, val_loader, save_path=args.save_dir,
                      learning_rate=args.lr, num_epochs=args.epochs)
    trainer.train()


if __name__ == "__main__":
    args = _parse_args()
    main(args)
