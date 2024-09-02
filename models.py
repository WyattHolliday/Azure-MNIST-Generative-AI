import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7), # Initial dense layer to expand latent vector
            # nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, 7, 7)), # Reshape to 128x7x7

            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), # 128x7x7 -> 128x14x14
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), # 128x14x14 -> 128x28x28
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 1, kernel_size=1, stride=1), # 128x28x28 -> 1x28x28
            nn.Tanh()  # Output activation function
        )

        # self.model = nn.Sequential(
        #     nn.Linear(latent_dim, 256),
        #     nn.ReLU(True),
        #     nn.Linear(256, 512),
        #     nn.ReLU(True),
        #     nn.Linear(512, 1024),
        #     nn.ReLU(True),
        #     nn.Linear(1024, 28*28),
        #     nn.Tanh()
        # )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2), # 1x28x28 -> 32x28x28
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32x28x28 -> 32x14x14

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 32x14x14 -> 64x14x14
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64x14x14 -> 64x7x7

            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # self.model = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2), # 1x28x28 -> 32x28x28
        #     nn.MaxPool2d(kernel_size=2, stride=2), # 32x28x28 -> 32x14x14
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), # 32x14x14 -> 64x14x14
        #     nn.MaxPool2d(kernel_size=2, stride=2), # 64x14x14 -> 64x7x7
        #     nn.ReLU(True),
        #     nn.Flatten(),
        #     nn.Linear(64*7*7, 256),
        #     nn.ReLU(True),
        #     nn.Linear(256, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, img):
        validity = self.model(img)
        return validity