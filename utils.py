import torchvision.transforms as transforms
# from azureml.opendatasets import MNIST
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch

def get_data_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize between -1 and 1
    ])
    dataset = MNIST(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def plot(data, x_label, y_label, title):
    plt.plot(data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def generate_img(generator, discriminator, num_images=1, num_row=5):
    gen_imgs = torch.empty(0)
    for i in range(num_images):
        pass_test = False
        while not pass_test:
            
            # Generate random image
            rand_tensor = torch.randn(1, 100)
            generator.eval()
            with torch.no_grad():
                gen_img = generator(rand_tensor)

            # Check if generated image passes discriminator test
            if discriminator is None:
                pass_test = True
            else:
                discriminator.eval()
                with torch.no_grad():
                    validity = discriminator(gen_img)
                    # If pass, exit loop
                    if validity.item() > 0.5:
                        pass_test = True

        # Save image
        gen_imgs = torch.cat((gen_imgs, gen_img), dim=0)

    save_image(gen_imgs.data, f'outputs/generated.png', nrow=num_row, normalize=True)