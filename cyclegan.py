import glob
import os
import random
from itertools import chain
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as utils
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

ncfl = 64          # Число каналов первого слоя
img_width = 256    # Ширина входного изображения 256
img_height = 256   # Высота входного изображения
RESCALE_SIZE = 256 # Размер рескейла в аугментациях
img_channel = 3    # RGB формат

"""Загрузка данных"""

class ImageDataset(Dataset):
    def __init__(self, root, unaligned=False, mode=None):
        self.unaligned = unaligned
        self.mode = mode

        self.files_A = sorted(glob.glob(os.path.join(root, f"{mode}A") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, f"{mode}B") + "/*.*"))

    def __getitem__(self, index):
        # Аугментации
        transform = {
            'train': transforms.Compose([
                transforms.Resize(int(RESCALE_SIZE * 1.12), Image.BICUBIC),
                transforms.RandomCrop(RESCALE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(nn.ModuleList(
                    [transforms.ColorJitter(brightness=0.4,
                                            contrast=0.4,
                                            saturation=0.4, hue=0.2), ]),p=0.10),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'test': transforms.Compose([
                transforms.Resize(size=(RESCALE_SIZE, RESCALE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        }

        item_A = transform[self.mode](Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = transform[self.mode](Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = transform[self.mode](Image.open(self.files_B[index % len(self.files_B)]))

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


"""### CycleGAN: Generator, Discriminator"""


class Generator(nn.Module):
    def __init__(self):
        self.ncfl = 64
        super(Generator, self).__init__()

        def conv_layer_relu(self, in_c, out_c, kernel_size, stride, padding):
            conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size, stride, padding),
                nn.InstanceNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
            return conv

        def transform_block(self, channels):
            tb = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(channels, channels, kernel_size=3),
                nn.InstanceNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(channels, channels, kernel_size=3),
                nn.InstanceNorm2d(channels)
            )
            return tb

        def transpose_conv_layer(self, in_c, out_c):
            conv = nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
            return conv

        self.first = nn.ReflectionPad2d(3)
        self.enc_1 = conv_layer_relu(self, 3, ncfl, 7, 1, 0)
        self.enc_2 = conv_layer_relu(self, ncfl, ncfl * 2, 3, 2, 1)
        self.enc_3 = conv_layer_relu(self, ncfl * 2, ncfl * 4, 3, 2, 1)

        # transformer
        self.tb_1 = transform_block(self, ncfl * 4)
        self.tb_2 = transform_block(self, ncfl * 4)
        self.tb_3 = transform_block(self, ncfl * 4)
        self.tb_4 = transform_block(self, ncfl * 4)
        self.tb_5 = transform_block(self, ncfl * 4)
        self.tb_6 = transform_block(self, ncfl * 4)

        # decoder
        self.dec_1 = transpose_conv_layer(self, ncfl * 4, ncfl * 2)
        self.dec_2 = transpose_conv_layer(self, ncfl * 2, ncfl)
        self.last = nn.Sequential(nn.ReflectionPad2d(3),
                                  nn.Conv2d(ncfl, 3, 7),
                                  nn.Tanh())

    def forward(self, x):
        # encoder
        x = self.first(x)
        x = self.enc_1(x)
        x = self.enc_2(x)
        x = self.enc_3(x)

        # transformer (residual blocks)
        x = self.tb_1(x) + x
        x = self.tb_2(x) + x
        x = self.tb_3(x) + x
        x = self.tb_4(x) + x
        x = self.tb_5(x) + x
        x = self.tb_6(x) + x

        # decoder
        x = self.dec_1(x)
        x = self.dec_2(x)
        output = self.last(x)

        return output


class Discriminator(nn.Module):
    def __init__(self):
        self.ncfl = 64
        super(Discriminator, self).__init__()

        def conv_layer_leaky_relu(self, in_c, out_c):
            conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )
            return conv

        self.first = nn.Sequential(nn.Conv2d(3, ncfl, 4, stride=2, padding=1),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.layer_1 = conv_layer_leaky_relu(self, ncfl, ncfl * 2)
        self.layer_2 = conv_layer_leaky_relu(self, ncfl * 2, ncfl * 4)
        self.layer_3 = conv_layer_leaky_relu(self, ncfl * 4, ncfl * 8)
        self.last = nn.Conv2d(ncfl * 8, 1, 4, padding=1)

    def forward(self, x):
        x = self.first(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.last(x)
        x = F.avg_pool2d(x, x.size()[2:])
        output = torch.flatten(x, 1)

        return output


"""### Упрощённое вычисление ошибки дискриминатора"""

# Создадим "резервуар" для заполнения его фейковыми изоюражениями с последующей
# случайной заменой старого на новое

class ImagePool:
    def __init__(self, pool_size=50):
        assert (pool_size > 0)
        self.pool_size = pool_size
        self.data = []

    def put_and_replace(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.pool_size:
                self.data.append(element)
                to_return.append(element)
            else:
                p = random.random()
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    to_return.append(self.data[random_id].clone())
                    self.data[random_id] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


"""Train"""

DATA_DIR = Path('project/cezanne2photo/')

def train_gan(dataset_dir, epochs, batch_size, lr=2e-4):
    # Создадим директории для хранения выходных файлов и весов
    try:
        os.makedirs("/content/drive/MyDrive/outputs")

    except OSError:
        pass


    out = Path("/content/drive/MyDrive/outputs")


    try:
        os.makedirs(os.path.join(out, "cezanne2photo", "A"))
        os.makedirs(os.path.join(out, "cezanne2photo", "B"))
    except OSError:
        pass

    try:
        os.makedirs(os.path.join(out, "weights"))
    except OSError:
        pass

    # Dataset

    dataset = ImageDataset(root=dataset_dir, unaligned=True, mode='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Создание модели
    netG_A2B = Generator().to(device)
    netG_B2A = Generator().to(device)
    netD_A = Discriminator().to(device)
    netD_B = Discriminator().to(device)

    # Дообучение (по необходимости)
    # netG_A2B.load_state_dict(torch.load(f'{out}/weights/cezanne2photo/netG_A2B_epoch_7.pth'))
    # netG_B2A.load_state_dict(torch.load(f'{out}/weights/cezanne2photo/netG_B2A_epoch_7.pth'))
    # netD_A.load_state_dict(torch.load(f'{out}/weights/cezanne2photo/netD_A_epoch_7.pth'))
    # netD_B.load_state_dict(torch.load(f'{out}/weights/cezanne2photo/netD_B_epoch_7.pth'))

    # Используемые лосс-функции
    cycle_loss = nn.L1Loss().to(device)
    identity_loss = nn.L1Loss().to(device)
    adversarial_loss = nn.MSELoss().to(device)

    # Оптимизаторы и lr-schedulers
    optimizer_G = optim.Adam(chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=lr, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))

    sched_G = optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.95)
    sched_D_A = optim.lr_scheduler.ExponentialLR(optimizer_D_A, gamma=0.95)
    sched_D_B = optim.lr_scheduler.ExponentialLR(optimizer_D_B, gamma=0.95)

    g_losses = []
    d_losses = []

    identity_losses = []
    gan_losses = []
    cycle_losses = []

    image_pool_A = ImagePool()
    image_pool_B = ImagePool()

    for epoch in range(8, epochs):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, data in progress_bar:
            # получим batch size данные
            real_image_A = data["A"].to(device)
            real_image_B = data["B"].to(device)
            batch_size = real_image_A.size(0)

            # real data label - 1, fake data label - 0
            real_label = torch.full((batch_size, 1), 1, device=device, dtype=torch.float32)
            fake_label = torch.full((batch_size, 1), 0, device=device, dtype=torch.float32)

            # ---------------------------------------
            # Обновление генераторов G_A2B и G_B2A
            # ---------------------------------------

            # Обнулим градиенты G_A and G_B
            optimizer_G.zero_grad()

            # Identity loss
            # G_B2A(A) должен быть равен A, если подаётся А
            identity_image_A = netG_B2A(real_image_A)
            loss_identity_A = identity_loss(identity_image_A, real_image_A) * 5.0
            # G_A2B(B) должен быть равен B, если подаётся B
            identity_image_B = netG_A2B(real_image_B)
            loss_identity_B = identity_loss(identity_image_B, real_image_B) * 5.0

            # GAN loss
            # GAN loss D_A(G_A(A))
            fake_image_A = netG_B2A(real_image_B)
            fake_output_A = netD_A(fake_image_A)
            loss_GAN_B2A = adversarial_loss(fake_output_A, real_label)
            # GAN loss D_B(G_B(B))
            fake_image_B = netG_A2B(real_image_A)
            fake_output_B = netD_B(fake_image_B)
            loss_GAN_A2B = adversarial_loss(fake_output_B, real_label)

            # Cycle loss
            recovered_image_A = netG_B2A(fake_image_B)
            loss_cycle_ABA = cycle_loss(recovered_image_A, real_image_A) * 10.0

            recovered_image_B = netG_A2B(fake_image_A)
            loss_cycle_BAB = cycle_loss(recovered_image_B, real_image_B) * 10.0

            # Комбинированный loss
            errG = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

            # Вычисление градиентов для G_A и G_B
            errG.backward()
            # Обновление весов G_A и G_B
            optimizer_G.step()

            if epoch > 99:
                sched_G.step()
            # ----------------------------------
            # Обновление дискриминатора D_A
            # ----------------------------------

            # Обнуление градиентов D_A 
            optimizer_D_A.zero_grad()

            # Real A image loss
            real_output_A = netD_A(real_image_A)
            errD_real_A = adversarial_loss(real_output_A, real_label)

            # Fake A image loss
            fake_image_A = image_pool_A.put_and_replace(fake_image_A)
            fake_output_A = netD_A(fake_image_A.detach())
            errD_fake_A = adversarial_loss(fake_output_A, fake_label)

            # Комбинированный loss
            errD_A = (errD_real_A + errD_fake_A) / 2

            # Вычисление градиентов D_A
            errD_A.backward()
            # Обновление весов D_A
            optimizer_D_A.step()

            if epoch > 99:
                sched_D_A.step()

            # -----------------------------------
            # Обновление дискриминатора D_B
            # -----------------------------------

            # Обнуление градиентов D_B
            optimizer_D_B.zero_grad()

            # Real B image loss
            real_output_B = netD_B(real_image_B)
            errD_real_B = adversarial_loss(real_output_B, real_label)

            # Fake B image loss
            fake_image_B = image_pool_B.put_and_replace(fake_image_B)
            fake_output_B = netD_B(fake_image_B.detach())
            errD_fake_B = adversarial_loss(fake_output_B, fake_label)

            # Комбинированный лосс
            errD_B = (errD_real_B + errD_fake_B) / 2

            # Вычисление градиентов для D_B
            errD_B.backward()
            # Обновление весов D_B
            optimizer_D_B.step()

            if epoch > 99:
                sched_D_B.step()

            progress_bar.set_description(
                f"[{epoch}/{epochs - 1}][{i}/{len(dataloader) - 1}] "
                f"Loss_D: {(errD_A + errD_B).item():.4f} "
                f"Loss_G: {errG.item():.4f} "
                f"Loss_G_identity: {(loss_identity_A + loss_identity_B).item():.4f} "
                f"loss_G_GAN: {(loss_GAN_A2B + loss_GAN_B2A).item():.4f} "
                f"loss_G_cycle: {(loss_cycle_ABA + loss_cycle_BAB).item():.4f}")

            if i % 100 == 0:
                utils.save_image(real_image_A,
                                 f"{out}/cezanne2photo/A/real_samples.jpg",
                                 normalize=True)
                utils.save_image(real_image_B,
                                 f"{out}/cezanne2photo/B/real_samples.jpg",
                                 normalize=True)

                fake_image_A = 0.5 * (netG_B2A(real_image_B).data + 1.0)
                fake_image_B = 0.5 * (netG_A2B(real_image_A).data + 1.0)

                utils.save_image(fake_image_A.detach(),
                                 f"{out}/cezanne2photo/A/fake_samples_epoch_{epoch}.jpg",
                                 normalize=True)
                utils.save_image(fake_image_B.detach(),
                                 f"{out}/cezanne2photo/B/fake_samples_epoch_{epoch}.jpg",
                                 normalize=True)

        # Промежуточное сохранение весов сетей
        torch.save(netG_A2B.state_dict(), f"{out}/weights/netG_A2B_epoch_{epoch}.pth")
        torch.save(netG_B2A.state_dict(), f"{out}/weights/netG_B2A_epoch_{epoch}.pth")
        torch.save(netD_A.state_dict(), f"{out}/weights/netD_A_epoch_{epoch}.pth")
        torch.save(netD_B.state_dict(), f"{out}/weights/netD_B_epoch_{epoch}.pth")

    # Финальное сохранение весов сетей
    torch.save(netG_A2B.state_dict(), f"{out}weights/netG_A2B.pth")
    torch.save(netG_B2A.state_dict(), f"{out}weights/netG_B2A.pth")
    torch.save(netD_A.state_dict(), f"{out}weights/netD_A.pth")
    torch.save(netD_B.state_dict(), f"{out}weights/netD_B.pth")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.cuda.empty_cache()

# Вызов функции train_gan
# train_gan(DATA_DIR, epochs=200, batch_size=7, lr=2e-4)

"""Test"""


def eval_gan(dataset_dir, batch_size):
    # Создание директории для результатов   
    try:
        os.makedirs('/content/drive/MyDrive/outputs/resuslts')
    except OSError:
        pass

    out = Path('/content/drive/MyDrive/outputs/resuslts')

    try:
        os.makedirs(os.path.join(out, "cezanne2photo", "A"))
        os.makedirs(os.path.join(out, "cezanne2photo", "B"))
    except OSError:
        pass

    # Dataset
    dataset = ImageDataset(root=dataset_dir, unaligned=True, mode='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Создание модели
    netG_A2B = Generator().to(device)
    netG_B2A = Generator().to(device)

    # Загрузка весов обученной в train_gan модели
    netG_A2B.load_state_dict(
        torch.load(os.path.join("/content/drive/MyDrive/outputs/weights", "cezanne2photo", "netG_A2B.pth")))
    netG_B2A.load_state_dict(
        torch.load(os.path.join("/content/drive/MyDrive/outputs/weights", "cezanne2photo", "netG_B2A.pth")))

    # Переключение модели в режим eval()
    netG_A2B.eval()
    netG_B2A.eval()

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for i, data in progress_bar:
        # get batch size data
        real_images_A = data["A"].to(device)
        real_images_B = data["B"].to(device)

        # Generate output
        fake_image_A = 0.5 * (netG_B2A(real_images_B).data + 1.0)
        fake_image_B = 0.5 * (netG_A2B(real_images_A).data + 1.0)

        # Save image files
        utils.save_image(fake_image_A.detach(), f"{out}/cezanne2photo/A/{i + 1:04d}.jpg", normalize=True)
        utils.save_image(fake_image_B.detach(), f"{out}/cezanne2photo/B/{i + 1:04d}.jpg", normalize=True)

        progress_bar.set_description(f"Process images {i + 1} of {len(dataloader)}")


torch.cuda.empty_cache()

# Запуск функции evan_gan
# eval_gan(DATA_DIR, batch_size)
