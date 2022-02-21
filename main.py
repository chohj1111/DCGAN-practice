from __future__ import print_function

#%matplotlib inline
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import models
from IPython.display import HTML


def parse_args():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument(
        "--dataroot",
        type=str,
        default="/home/irteam/cajun7-dcloud-dir/DCGAN-practice/data/",
        help="the path to the directory containing the data.",
    )

    # Data loader settings
    parser.add_argument(
        "--workers", type=int, default=2, help="the number of workers for dataloader."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="the number of samples for a batch"
    )

    # Model settings
    parser.add_argument(
        "--image_size", type=int, default=64, help="the size of an image."
    )
    parser.add_argument(
        "--channel_dim",
        type=int,
        default=3,
        help="Number of channels in the training images.",
    )
    parser.add_argument(
        "--z_dim",
        type=int,
        default=100,
        help="size of z latent vector (i.e. size of generator input)",
    )
    parser.add_argument(
        "--g_feature_dim",
        type=int,
        default=64,
        help="size of feature maps in generator ",
    )
    parser.add_argument(
        "--d_feature_dim",
        type=int,
        default=64,
        help="size of feature maps in discriminator ",
    )

    # Trainer settings
    parser.add_argument(
        "--n_gpu", type=int, default=1, help="the number of gpus to be used."
    )
    parser.add_argument(
        "--n_epochs", type=int, default=5, help="the number of training epochs."
    )

    # Optimization
    parser.add_argument("--lr", type=float, default=2e-4, help="the learning rate.")

    parser.add_argument(
        "--beta1", type=float, default=0.5, help="Beta1 hyperparam for Adam optimizers."
    )

    args = parser.parse_args()
    return args


def imshow_grid(img):
    img = torch.utils.make_grid(img.cpu().detach())
    img = (img + 1) / 2
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main():
    # parse arguments
    args = parse_args()

    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = dset.ImageFolder(root=args.dataroot, transform=preprocess)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    print(len(dataloader))
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and args.n_gpu > 0) else "cpu"
    )

    real_image = next(iter(dataloader))
    # imshow_grid(real_image.view(-1, 3, args.image_size, args.image_size))

    netG = models.G(args).to(device)
    if (device.type == "cuda") and (args.n_gpu > 1):
        netG = nn.Dataparallel(netG, list(range(args.n_gpu)))
    netG.apply(weights_init)

    netD = models.D(args).to(device)
    if (device.type == "cuda") and (args.n_gpu > 1):
        netD = nn.Dataparallel(netD, list(range(args.n_gpu)))
    netD.apply(weights_init)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, args.z_dim, 1, 1, device=device)

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    img_list = []
    G_loss_list = []
    D_loss_list = []
    iters = 0

    print("Training loop started")

    for epoch in range(args.n_epochs):
        for i, data in enumerate(dataloader):
            #################
            netD.zero_grad()

            # data_ = data[0].reshape(-1, args.image_size * args.image_size)
            noise = torch.randn(args.batch_size, args.z_dim, 1, 1, device=device)

            p_real = netD(data[0].to(device))
            p_fake = netD(netG(noise))

            D_x = p_real.mean().item()
            D_G_z1 = p_fake.mean().item()

            loss_d = criterion(
                p_real, torch.ones(p_real.size()).to(device)
            ) + criterion(p_fake, torch.zeros(p_fake.size()).to(device))

            loss_d.backward()
            optimizerD.step()

            ##################
            netG.zero_grad()

            p_fake = netD(netG(noise))
            loss_g = criterion(p_fake, torch.ones(p_fake.size()).to(device))

            D_G_z2 = p_fake.mean().item()

            loss_g.backward()
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print(
                    "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                    % (
                        epoch,
                        args.n_epochs,
                        i,
                        len(dataloader),
                        loss_d.item(),
                        loss_g.item(),
                        D_x,
                        D_G_z1,
                        D_G_z2,
                    )
                )

            # Save Losses for plotting later
            G_loss_list.append(loss_g.item())
            D_loss_list.append(loss_d.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or (
                (epoch == args.n_epochs - 1) and (i == len(dataloader) - 1)
            ):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_loss_list, label="G")
    plt.plot(D_loss_list, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Visualize G
    #%%capture
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(
        fig, ims, interval=1000, repeat_delay=1000, blit=True
    )

    # HTML(ani.to_jshtml())

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                real_batch[0].to(device)[:64], padding=5, normalize=True
            ).cpu(),
            (1, 2, 0),
        )
    )

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    main()

