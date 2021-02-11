from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import torchvision.transforms as transforms
import torchvision.models as models

import copy


imsize = 128

loader = transforms.Compose([transforms.Resize(imsize),  
                             transforms.CenterCrop(imsize),
                             transforms.ToTensor()])



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Отрисовка изображения"""

# def image_loader(image_name):
#     image = Image.open(image_name)
#     image = loader(image).unsqueeze(0)
#     return image.to(device, torch.float)
#
#
# style_img = image_loader("/content/drive/MyDrive/images/picasso.jpg")
# content_img = image_loader("/content/drive/MyDrive/my_images/marx.jpg")
#
# """## Вывод используемых изображений на экран"""
#
# unloader = transforms.ToPILImage() # тензор в кратинку


""" Создаём лоссы"""

class ContentLoss(nn.Module):

        def __init__(self, target,):
            super(ContentLoss, self).__init__()
            self.target = target.detach()                 #это константа. Убираем ее из дерева вычеслений
            self.loss = F.mse_loss(self.target, self.target) # для инициализации с некоторыми аргументами

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input

def gram_matrix(input):
        batch_size, f_map_num, h, w = input.size()

        features = input.view(batch_size * h, w * f_map_num)  # преобразование ФМ под матрицу Грама

        G = torch.mm(features, features.t())  # вычисление "скалярного" произведения (матрицы Грама)

        # нормируем значения матрицы Грама,
        # разделив на число элементов каждой фичимапы
        return G.div(batch_size * h * w * f_map_num)

class StyleLoss(nn.Module):
        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()
            self.loss = F.mse_loss(self.target, self.target)

        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input

""" Нормализация используемых изображений """

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            # .view the mean and std to make them [C x 1 x 1] so that they can
            # directly work with image Tensor of shape [B x C x H x W].
            # B is batch size. C is number of channels. H is height and W is width.
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            # нормализация изображения
            return (img - self.mean) / self.std

""" Создание архитектуры NST на основе предобученной VGG19 """

#Введем переменные для перебора слоёв, чтобы ввести слои с лоссами
content_layers_default = ['conv_4']       
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

cnn = models.vgg19(pretrained=True).features.to(device).eval()

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                   style1_img, content_img,
                                   content_layers=content_layers_default,
                                   style_layers=style_layers_default):
        cnn = copy.deepcopy(cnn)

        # нормализация
        normalization = Normalization(normalization_mean, normalization_std).to(device)

        # для итерируемого доступа к content/syle losses создаем списки
        content_losses = []
        style_losses = []

        # функция добавления style losses в виде слоя в архитектуру нейросети
        def add_style_losses(style_img):
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

        # т.к. cnn - это nn.Sequential, то также возьмем nn.Sequential,
        # чтобы сделать свою кастомную сборку модели на базе vgg16
        model = nn.Sequential(normalization)

        i = 0  # инкримент при каждом встречном conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                #Переопределим out-of-place (not in-place) relu уровень
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                # добавление content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # добавление style losses:
                add_style_losses(style1_img)


        #выбрасываем все уровни после последенего style loss или content loss
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

""" Функция переноса стиля """

def get_input_optimizer(input_img):
        #добоваляет содержимое тензора катринки в список изменяемых оптимизатором параметров
        optimizer = optim.LBFGS([input_img.requires_grad_()]) 
        return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                        content_img, style_img, input_img, num_steps,
                        style_weight=10000, content_weight=1):
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = get_style_model_and_losses(cnn,
            normalization_mean, normalization_std, style_img, content_img)
        optimizer = get_input_optimizer(input_img)

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()

                model(input_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                
                #взвешивание ошибки
                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        # последняя корректировка
        input_img.data.clamp_(0, 1)

        return input_img


# input_img = content_img.clone() # для входного изображения вместо шума используем контент для ускорения процесса
#
# output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
#                             content_img, style1_img, input_img)

""" Отрисовка outputs для переноса нескольких стилей на 4 участка контент-фото"""

# output_1 = output.to('cpu').squeeze().permute(1,2,0).detach()
#
# fig, axs = plt.subplots(3, 2, figsize = (10, 14))
# axs[0, 0].imshow(style1_img.squeeze(0).permute(1,2,0), aspect="auto")
# axs[0, 0].set_title('Style1 Image', fontsize = 20)
# axs[0, 0].axis('off')
# axs[0, 1].imshow(style2_img.squeeze(0).permute(1,2,0), aspect="auto")
# axs[0, 1].set_title('Style2 Image', fontsize = 20)
# axs[0, 1].axis('off')
# axs[1, 0].imshow(style3_img.squeeze(0).permute(1,2,0), aspect="auto")
# axs[1, 0].set_title('Style3 Image', fontsize = 20)
# axs[1, 0].axis('off')
# axs[1, 1].imshow(style4_img.squeeze(0).permute(1,2,0), aspect="auto")
# axs[1, 1].set_title('Style4 Image', fontsize = 20)
# axs[1, 1].axis('off')
# axs[2, 0].imshow(content_img.squeeze(0).permute(1,2,0), aspect="auto")
# axs[2, 0].set_title('Content Image', fontsize = 20)
# axs[2, 0].axis('off')
# axs[2, 1].imshow(output_1, aspect="auto")
# axs[2, 1].set_title('Output Image', fontsize = 20)
# axs[2, 1].axis('off')
#
# plt.ioff()
# plt.show()

