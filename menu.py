
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
button_1 = InlineKeyboardButton('CUSTOM', callback_data='b1')
button_2 = InlineKeyboardButton('Paul Cézanne', callback_data='b2')

keyboard = InlineKeyboardMarkup(row_width=2).add(button_1, button_2)

button_3 = InlineKeyboardButton('Отмена', callback_data='cancel')
cancel_board = InlineKeyboardMarkup().insert(button_3)