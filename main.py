import io
import logging

import aiohttp
from PIL import Image
from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text
from aiogram.types.message import ContentType
from aiogram.utils.markdown import text

from config import TOKEN
from menu import menu
import nst

from math import sqrt

#import util
#import torch

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())




#netG = util.load_esr_model('weight/esr.pth')
#netG.eval()


@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    """
    Ответ пользователю на команды `/start`
    """
    if message:
        await message.answer('Салам, братан!\nВыбери фичу и кидай пикчу - посмотришь, что выйдет!:)\n'
                             'Подробная инфа по команде /help\n', reply_markup=menu)


@dp.message_handler(text='NST')
async def get_nst(message: types.Message):
    await message.answer('Ща затюнингую нормально!')

@dp.message_handler(text='SRGAN')
async def get_srgan(message: types.Message):
    await message.answer('Ща будет всё чотка!')




# @dp.message_handler(Text(equals=['NST', 'SRGAN']))
# async def get_mode(message: types.Message):
#     await m



@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    """
    Ответ пользователю на команды `/help`
    """
    await message.reply("NST - переносит стиль\n"
                        "SRGAN - улучшает разрешение")


@dp.message_handler(content_types=types.ContentTypes.DOCUMENT | types.ContentTypes.PHOTO)
async def get_photo_or_doc(message):

    if message.document:
        ref = await message.document.get_url()
    else:
        ref = await message.photo[-1].get_url()


    async with aiohttp.ClientSession() as cs:
        async with cs.get(ref) as reaction:
            img = Image.open(io.BytesIO(await reaction.read()))

    #image.save("temp/" + f"{message.from_user['username']}_{datetime.now().strftime('%H:%M:%S')}" + ".jpg", "JPEG")


    if img.width * img.height > 256 * 256:
        await waiting(message)
        ratio = sqrt(img.width * img.height / (256 * 256))
        img = img.resize((round(img.width / ratio), round(img.height / ratio)), Image.BICUBIC)
        #await return_image(message, img, "Уай, уай, мощи не хватает на бОльшее!")

    lr = transforms.ToTensor()(image).unsqueeze(0)
    #fake = netG(lr).clamp_(0., 1)
    #image = transforms.ToPILImage(mode='RGB')(fake[0])
    await return_image(message, img, 'Хвала Аллаху, получилось!')


async def return_image(message: types.Message, image: Image, text: str):
    bytes = io.BytesIO()
    bytes.name = 'image.jpeg'
    image.save(bytes, 'JPEG')
    bytes.seek(0)
    await message.reply_photo(bytes, caption=text)


async def waiting(message: types.Message):
    await message.answer("Эу, эу, не наводи суету, ща всё будет!")


@dp.message_handler(content_types=ContentType.ANY)
async def unknown_message(msg: types.Message):
    message_text = text("Братишка, я хз чо это.\nСмотри тут --> '/help'")
    await msg.reply(message_text)

if __name__ == '__main__':
    executor.start_polling(dp)

