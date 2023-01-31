# Speech Commands dataset excerpt

This is a small excerpt of the [Speech Commands Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) for use in a tutorial on tensorflow.org. Please refer to the original [dataset](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz) for documentation and license information."""

mini_speech_commands — пакет из 8000 примеров звуковых файлов *.wav, разбитых по 8 классам (значениям слов). В каждом классе 1000 примеров.
Датасет взят со страницы учебника TensorFlow: https://www.tensorflow.org/tutorials/audio/simple_audio

msc_dataset.json — Файл имен файлов со звуками, структурированных по классам.
Этот файл нужен для возможности пакетной загрузки датасетов при работе в браузере на стороне клиента без использования node.js.

Если файла msc_dataset.json нет, то его необходимо создать, запустив скрипт ./node-files-list-creator.js в node.js.
После создания файла нужно пересобрать проект, чтобы msc_dataset.json попал в новую сборку (bundle).