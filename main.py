import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from models import Generator, Discriminator
from utils import get_data_loader, plot, generate_img
import timeit

def train(generator, discriminator, params=None):
    # Default hyperparameters
    if params is None:
        params = {
            "latent_dim": 100,
            "batch_size": 64,
            "gen_learning_rate": 0.00005,
            "dis_learning_rate": 0.00005,
            "num_epochs": 100
        }
    
    # Data loader
    dataloader = get_data_loader(params["batch_size"])

    # optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=params["gen_learning_rate"])
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=params["dis_learning_rate"])

    # Loss function
    adversarial_loss = nn.BCELoss()

    # Loss Statistics
    gen_losses = []
    dis_losses = []

    times = []
    
    # Training Loop
    generator.train()
    discriminator.train()
    for epoch in range(1, params["num_epochs"] + 1):
        start_time = timeit.default_timer()
        for i, (imgs, _) in enumerate(dataloader):
            ones = torch.ones(imgs.size(0), 1)
            zeros = torch.zeros(imgs.size(0), 1)

            # Train Generator
            gen_optimizer.zero_grad()
            rand_tensor = torch.randn(imgs.size(0), params["latent_dim"])
            gen_imgs = generator(rand_tensor)
            gen_loss = adversarial_loss(discriminator(gen_imgs), ones)
            gen_loss.backward()
            gen_optimizer.step()
            gen_losses.append(gen_loss.item())

            # Train Discriminator
            dis_optimizer.zero_grad()
            real_loss = adversarial_loss(discriminator(imgs), ones)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), zeros)
            dis_loss = (real_loss + fake_loss) / 2
            dis_loss.backward()
            dis_optimizer.step()
            dis_losses.append(dis_loss.item())

            # Calculate percentage of fake pictures found not to be fake
            fake_predictions = discriminator(gen_imgs.detach())
            fake_accuracy = (fake_predictions < 0.5).float().mean().item() * 100

            # Calculate percentage of real pictures found to be real
            real_predictions = discriminator(imgs)
            real_accuracy = (real_predictions >= 0.5).float().mean().item() * 100

            print(f"[Epoch {epoch}/{params['num_epochs']}] [Batch {i+1}/{len(dataloader)}] [D loss: {round(dis_loss.item(), 3)}] [G loss: {round(gen_loss.item(), 2)}] [Fake Acc: {round(fake_accuracy, 2)}%] [Real Acc: {round(real_accuracy, 2)}%]")

        # Save generated images at the end of each epoch
        save_image(gen_imgs.data[:25], f'outputs/{epoch}.png', nrow=5, normalize=True)

        # Save model checkpoints
        torch.save(generator.state_dict(), f'models/generators/generator_epoch_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'models/discriminators/discriminator_epoch_{epoch}.pth')

        # Calculate time taken for epoch
        time = timeit.default_timer() - start_time
        times.append(time)
        print(f"Time taken for epoch {epoch}: {round(time / 60, 1)} minutes")
    
    # Plot Losses
    plot(gen_losses, "Iterations", "Loss", "Generator Loss")
    plot(dis_losses, "Iterations", "Loss", "Discriminator Loss")

    # Total time
    total_time = sum(times)
    print(f"Total time taken: {round(total_time / 3600, 2)} hours")

def main():
    params = {
        "latent_dim": 100,
        "batch_size": 64,
        "gen_learning_rate": 0.00005,
        "dis_learning_rate": 0.00005,
        "num_epochs": 60
    }
    # Model and Optimizer
    generator = Generator(params["latent_dim"])
    generator.load_state_dict(torch.load('models/generators/generator_epoch_60.pth', weights_only=True))
    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load('models/discriminators/discriminator_epoch_60.pth', weights_only=True))

    # Train
    # train(generator, discriminator, params)

    # Generate Image
    generate_img(generator, discriminator, 100, num_row=10)

if __name__ == "__main__":
    main()