
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

"""
keyboard
"""

# "Физическая" клавиатура
# menu = ReplyKeyboardMarkup(
#     keyboard=[
#         [
#             KeyboardButton(text='/NST'),
#             KeyboardButton(text='/CycleGAN')
#         ],
#     ],
#     resize_keyboard=True
# )



"""
Inline keyboard 
"""

# Всплывающая клавиатура

# Меню
button_1 = InlineKeyboardButton('CUSTOM', callback_data='b10')
button_2 = InlineKeyboardButton('Paul Cézanne', callback_data='b20')


menu = InlineKeyboardMarkup(row_width=2).add(button_1, button_2)

# Отмена
button_3 = InlineKeyboardButton('Отмена❌', callback_data='cancel')
cancel_button = InlineKeyboardMarkup().insert(button_3)

# Продолжить
button_4 = InlineKeyboardButton('Продолжить', callback_data='b11')
continue_button1 = InlineKeyboardMarkup().add(button_4)

button_5 = InlineKeyboardButton('Продолжить', callback_data='b21')
continue_button2 = InlineKeyboardMarkup().add(button_5)

"""
Пояснение к выбору
callback_data: b - button, 
               1/2 - NST/GAN,
               0/1 - заходим впервые/продолжаем быть в режиме
"""
