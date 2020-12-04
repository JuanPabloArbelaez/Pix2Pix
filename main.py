from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from skimage import color
import numpy as np

from models import *
from visualize import show_tensor_images



# Training
adv_criterion = nn.BCEWithLogitsLoss()
recon_criterion = nn.L1Loss()
lambda_recon = 200
n_epochs = 300
input_dim = 3
real_dim = 3
display_step = 50
batch_size = 4
lr = 2e-4
target_shape = 256
device = "cuda"


# # Dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.ImageFolder("maps", transform=transform)

# Initialize generator and discriminator
gen = UNet(input_dim, real_dim).to(device)
disc = Discriminator(input_dim + real_dim).to(device)

# Optimizers
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(m.bias, 0)

gen = gen.apply(weights_init)
disc = disc.apply(weights_init)


def train(save_model=False):
    mean_gen_loss = 0
    mean_disc_loss = 0 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    cur_step = 0

    for epoch in range(n_epochs):
        for image, _ in tqdm(dataloader):
            image_width = image.shape[3]
            condition = image[:, :, :, :image_width // 2]
            condition = nn.functional.interpolate(condition, size=target_shape)
            real = image[:, :, :, image_width // 2:]
            real = nn.functional.interpolate(real, size=target_shape)
            cur_batch_size = len(condition)
            condition = condition.to(device)
            real = real.to(device)

            ### Update Discriminator ###
            disc_opt.zero_grad()
            with torch.no_grad():
                fake = gen(condition)
            disc_fake_hat = disc(fake.detach(), condition)
            disc_fake_loss = adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))
            disc_real_hat = disc(real, condition)
            disc_real_loss = adv_criterion(disc_real_hat, torch.zeros_like(disc_real_hat))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            ### Update Generator ###
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon)
            gen_loss.backward()
            gen_opt.step()

            # Keep track of avg disc loss 
            mean_disc_loss += disc_loss.item() / display_step
            # keep track of avg gen loss
            mean_gen_loss += gen_loss.item() / display_step

            ## Visualize ##
            if cur_step % display_step == 0:
                if cur_step > 0:
                    print(f"Epoch: {epoch}  Step: {cur_step}  Avg Gen loss: {mean_gen_loss}  Avg Disc loss: {mean_disc_loss}")
                else:
                    print("Pretrained Initial State")

                show_tensor_images(condition, size=(input_dim, target_shape, target_shape))
                show_tensor_images(real, size=(input_dim, target_shape, target_shape))
                show_tensor_images(fake, size=(input_dim, target_shape, target_shape))
                mean_gen_loss = 0

                if save_model:
                    torch.save(
                        {'gen': gen.state_dict(),
                        'gen_opt': gen_opt.state_dict(),
                        'disc': disc.state_dict(),
                        'disc_opt': disc_opt.state_dict(),
                        }, f"pix2pix_{cur_step}.pth")
            cur_step += 1


if __name__ == "__main__":
    train()
