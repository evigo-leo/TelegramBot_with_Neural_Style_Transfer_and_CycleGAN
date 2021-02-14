import io
import logging
from math import sqrt

import aiohttp
import torch
from PIL import Image
from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.types.message import ContentType
from aiogram.utils.markdown import text
from torchvision.transforms import transforms

from config import TOKEN
from cyclegan import Generator, device
from keyboards import menu, cancel_button, continue_button1, continue_button2
from nst import run_style_transfer, cnn, cnn_normalization_mean, cnn_normalization_std
from states import PhotoTransform

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())

# Импортируем модель класса Generator
gen = Generator().to(device)

# Выбор GPU или CPU
if torch.cuda.is_available():
    map_location = lambda storage, loc: storage.cuda()
else:
    map_location = 'cpu'

# Загрузка весов модели и устанавливаем инференс-режим
gen.load_state_dict(torch.load('weights/netG_B2A.pth', map_location=map_location))
gen.eval()


""" БАЗОВЫЕ КОМАНДЫ /"""

@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    """
    Ответ пользователю на команды `/start`
    """
    if message:
        await message.answer('Выбери фичу и кидай пикчу!😊\n'
                             'Подробности по команде /info\n', reply_markup=menu)


@dp.message_handler(commands=['info'])
async def process_help_command(message: types.Message):
    """
    Ответ пользователю на команды `/info`
    """
    await message.reply(
        "CUSTOM - переносит стиль/текстуры/цветовую гамму понравившегося изображения на контент-фото.\n\n"
        # "Команда /NST.\n\n"
        "Paul Cézanne - превращает фото в живописную картину в стиле Поля Сезанна.\n\n"
        # "Команда /CycleGAN.\n\n"
        "Контент-фото преобразуется к разрешению 256*256 с сохранением пропорции сторон.\n\n"
        "Для возврата в меню используйте команду /start")

""" РАБРОТА С ИНЛАЙН-КЛАВИАТУРОЙ"""

# Хендлер CallbackQuery для работы с инлайн-клавиатурами/кнопками
@dp.callback_query_handler(lambda c: c.data[0] == 'b')     # Фильтруем попадание по 1-му символу из 3-х callback_data
async def process_callback_keyboard(callback_query: types.CallbackQuery):

    mode_code = callback_query.data[1]                # 2й символ определяет режим
    continuation_mode = callback_query.data[-1]       # 3й - продолжить в этом режиме

    if mode_code == '1':  # отработка кнопки под NST
        if continuation_mode == '0':
            await bot.answer_callback_query(callback_query.id, text='Режим CUSTOM активирован!✅', show_alert=True)
        await bot.send_message(callback_query.from_user.id, 'Жду контент!', reply_markup=cancel_button)
        await PhotoTransform.PT1.set()  # Устанавливаем 1й стейт машины состояний


        # Исключение любых типов данных в этом состоянии кроме фото и док
        @dp.message_handler(content_types=ContentType.ANY ^ (ContentType.PHOTO | ContentType.DOCUMENT))
        async def unknown_message(message: types.Message):
            await message.reply('Надо завершить ❌ или продолжить')


        # Исключение любых типов данных в этом состоянии кроме фото и док
        @dp.message_handler(content_types=['text', 'video', 'sticker', 'audio', 'voice', 'unknown'],
                            state=PhotoTransform.PT1)
        async def incorrect_content(message: types.Message):
            await message.reply('Надо завершить ❌ или продолжить')


        # Этот хендлер действует, если вдруг передумали использовать один режим, а хотим другой
        @dp.callback_query_handler(text='cancel', state=PhotoTransform.PT1)
        async def cancel(callback_query: types.CallbackQuery, state: FSMContext):
            await callback_query.answer('Режим CUSTOM отменён!❌', show_alert=True)
            await callback_query.message.edit_reply_markup()
            await state.reset_state()  # сбрасываем стейт МС в случае отмены
            await bot.send_message(callback_query.from_user.id, 'Меню: /start')


    # отработка кнопки под CycleGAN
    else:
        if continuation_mode == '0':
            await bot.answer_callback_query(callback_query.id, text='Режим Paul Cézanne активирован!✅', show_alert=True)
        await bot.send_message(callback_query.from_user.id, 'Закидывай! Сейчас сделаю всё по красоте!', reply_markup=cancel_button)
        state = dp.current_state(user=callback_query.from_user.id)
        await state.set_state('GAN mode')


        # Исключение любых типов данных в этом состоянии кроме фото и док
        @dp.message_handler(content_types=['text', 'video', 'sticker', 'audio', 'voice', 'unknown'],
                            state='GAN mode')
        async def incorrect_content(message: types.Message):
            await message.reply('Надо завершить ❌ или продолжить')


        # в случае отмены CycleGAN
        @dp.callback_query_handler(text='cancel', state='GAN mode')
        async def cancel(callback_query: types.CallbackQuery, state: FSMContext):
            await callback_query.answer('Режим Paul Cézanne отменён!❌', show_alert=True)
            await callback_query.message.edit_reply_markup()
            await state.reset_state()
            await bot.send_message(callback_query.from_user.id, 'Меню: /start')


# код для работы через / (не инлайн-кнопки)
"""
@dp.message_handler(commands='NST')
async def get_nst(message: types.Message):
    await message.answer('Жду контент!', reply_markup=cancel_button)
    await PhotoTransform.PT1.set()

    @dp.callback_query_handler(text='cancel', state=PhotoTransform.PT1)
    async def cancel(callback_query: types.CallbackQuery, state: FSMContext):
        await callback_query.answer('Вы отменили текущий режим!❌', show_alert=True)
        await callback_query.message.edit_reply_markup()
        await state.reset_state()
        await bot.send_message(callback_query.from_user.id, 'Для возврата в меню команда /start')


@dp.message_handler(commands='CycleGAN')
async def get_cyclegan(message: types.Message):
    await message.answer('Закидывай! Сейчас сделаю всё по красоте', reply_markup=cancel_button)
    state = dp.current_state(user=message.from_user.id)
    await state.set_state('GAN mode')

    @dp.callback_query_handler(text='cancel', state='GAN mode')
    async def cancel(callback_query: types.CallbackQuery, state: FSMContext):
        await callback_query.answer('Вы отменили текущий режим!❌', show_alert=True)
        await callback_query.message.edit_reply_markup()
        await state.reset_state()
        await bot.send_message(callback_query.from_user.id, 'Для возврата в меню команда /start')
"""

""" ХЕНДЛЕРЫ С ЗАПУСКОМ НЕЙРОСЕТЕЙ"""

# Хендлеры по (1, 2, 3 ниже) обработке фото/документов
@dp.message_handler(content_types=types.ContentTypes.DOCUMENT | types.ContentTypes.PHOTO,
                    state=PhotoTransform.PT1)
async def get_photo_or_doc1(message, state: FSMContext):
    if message.document:  # Определяем фото или док
        ref = await message.document.get_url()
    else:
        ref = await message.photo[-1].get_url()


    async with aiohttp.ClientSession() as cs:
        async with cs.get(ref) as reaction:
            img = Image.open(io.BytesIO(await reaction.read()))  # Изображение в бинарном потоке


    if img.width * img.height > 256 * 256:  # преобразование входного изображения под 256*256
        ratio = sqrt(img.width * img.height / (256 * 256))  # c сохранением пропорции
        img = img.resize((round(img.width / ratio), round(img.height / ratio)), Image.BICUBIC)


    trans = transforms.ToTensor()
    content_img = trans(img).unsqueeze(0)  # фото -> тензор [1, 3, 256, 256]
    imsize = (content_img.size(2), content_img.size(3))
    await state.update_data(size=imsize)
    await state.update_data(pic=content_img)
    await message.answer('Теперь стиль...', reply_markup=cancel_button)
    await PhotoTransform.next()


    # Исключение любых типов данных в этом состоянии кроме фото и док
    @dp.message_handler(content_types=['text', 'video', 'sticker', 'audio', 'voice', 'unknown'],
                        state=PhotoTransform.PT1)
    async def incorrect_content(message: types.Message):
        await message.reply('Надо завершить ❌ или продолжить')


    # Хендлер под кнопку "Отмена" Если передумали использовать NST на этапе загрузки стиля
    @dp.callback_query_handler(text='cancel', state=PhotoTransform.PT2)
    async def cancel(callback_query: types.CallbackQuery, state: FSMContext):
        await callback_query.answer('Режим CUSTOM отменён!❌', show_alert=True)
        await callback_query.message.edit_reply_markup()
        await state.reset_state()
        await bot.send_message(callback_query.from_user.id, 'Меню: /start')


@dp.message_handler(content_types=types.ContentTypes.DOCUMENT | types.ContentTypes.PHOTO,
                    state=PhotoTransform.PT2)
async def get_photo_or_doc2(message, state: FSMContext):
    if message.document:
        ref = await message.document.get_url()
    else:
        ref = await message.photo[-1].get_url()


    async with aiohttp.ClientSession() as cs:
        async with cs.get(ref) as reaction:
            img = Image.open(io.BytesIO(await reaction.read()))


    data = await state.get_data()
    content_img = data.get('pic')
    imsize = data.get('size')
    input_img = content_img.clone()

    trans = transforms.Compose([transforms.Resize(imsize),
                                transforms.ToTensor()])
    style_img = trans(img).unsqueeze(0)  # преобразование фото стиля в соответствии с фото контента
    await waiting(message)  # ответ, сигнализирующий о том, что процесс генерации выполняется

    # Исполнение кода с нейросетью NST
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                               content_img, style_img, input_img, num_steps=200)

    output_1 = output.to('cpu').squeeze().detach()
    pic = transforms.ToPILImage()(output_1)  # Тензор -> фото
    await return_nst_image(message, pic, 'Вуаля! свершилось!🥳🎉\n'
                                     'Меню: /start.')
    await state.finish()  # Обнуление стостояний МС


# Аналогично в случае с CycleGAN
@dp.message_handler(content_types=types.ContentTypes.DOCUMENT | types.ContentTypes.PHOTO,
                    state='GAN mode')
async def get_photo_or_doc3(message, state: FSMContext):
    if message.document:
        ref = await message.document.get_url()
    else:
        ref = await message.photo[-1].get_url()


    async with aiohttp.ClientSession() as cs:
        async with cs.get(ref) as reaction:
            img = Image.open(io.BytesIO(await reaction.read()))


    if img.width * img.height > 256 * 256:
        ratio = sqrt(img.width * img.height / (256 * 256))
        img = img.resize((round(img.width / ratio), round(img.height / ratio)), Image.BICUBIC)


    await waiting(message)
    trans = transforms.ToTensor()
    tensor_img = trans(img).unsqueeze(0)
    tensor_img = gen(tensor_img)
    fake_img = 0.5 * (tensor_img.data + 1.0).detach()
    img = transforms.ToPILImage(mode='RGB')(fake_img[0])

    await return_cyclegan_image(message, img, "Вуаля! Свершилось!🥳🎉\n"
                                     "Меню: /start.")
    await state.reset_state()


"""ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ"""


# Возврат ботом готовой фотографии в ответ
async def return_nst_image(message: types.Message, image: Image, text: str):
    bytes = io.BytesIO()
    bytes.name = 'image.jpeg'
    image.save(bytes, 'JPEG')
    bytes.seek(0)
    await message.reply_photo(bytes, caption=text, reply_markup=continue_button1)


async def return_cyclegan_image(message: types.Message, image: Image, text: str):
    bytes = io.BytesIO()
    bytes.name = 'image.jpeg'
    image.save(bytes, 'JPEG')
    bytes.seek(0)
    await message.reply_photo(bytes, caption=text, reply_markup=continue_button2)


# Сообщение об ожидании выполнения задачи
async def waiting(message: types.Message):
    await message.answer("Процесс пошел, дело времени!🕑")


# Обработа всех команд и сообщений, неизвестных боту
@dp.message_handler(content_types=ContentType.ANY)
async def unknown_message(message: types.Message):
    message_text = text("Друже, я не розумiю, що це🤷‍♂️.\nДивись тут ➡️ /info")
    await message.reply(message_text)


# Запуск бота
if __name__ == '__main__':
    executor.start_polling(dp)