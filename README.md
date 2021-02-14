# TelegramBot_with_Neural_Style_Transfer_&_CycleGAN
PyTorch and aiogram implementation of telegram bot of NST &amp; CycleGAN

Описание файлов проекта.
1. Weights - папка с предобученными гиперпараметрами на датасете с картинами художника Paul Cézanne, в боте используются веса генератора netG_B2A.
2. .env - конфигурационный файл с токеном бота.
3. bot.conf - файл с командой запуска бота с возможностью автовосстановления после перезагрузки сервера.
4. config.py - файл с переменной окружения бота (сюда можно добавить, например, порты, протоколы, сертификаты для webhook).
5. cyclegan.py -   модель CycleGAN.
6. keyboards.py - инлайн-клавиатуры и кнопки для управления логикой бота.
7. nst.py - модель Neural Style Transfer.
8. requirements.txt - требования для библиотек, устанавливаемых на сервере AWS.
9. states.py - набор состояний, использумых машиной состояний (Finite State Machine), для функционирования логики бота.

Команды бота:
1) /start - запуск бота, вывод главного меню;
2) /info - функционал бота, его особенности;
3) кнопки "CUSTOM" и "Paul Cézanne" устанавливают режим работы NST и CycleGAN соответственно;
3) кнопка "Продолжить" сохраняет выбранный режим работы после получения результата.

Дополнительная информация:
- режим CUSTOM основан на NST, поэтому обучение происходит непостредственно в момент запуска, 
  время исполнения ~3-5 минут;
- режим Paul Cézanne использует уже обученный GAN (порядка 200 эпох с ~800 итераций на каждую),
  инференс не более нескольких секунд.
