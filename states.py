from aiogram.dispatcher.filters.state import StatesGroup, State

# Машина состояний для NST
class PhotoTransform(StatesGroup):
    PT1 = State()
    PT2 = State()
