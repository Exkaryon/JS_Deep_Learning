const tf = require('@tensorflow/tfjs');

// Размеры картинок в датасете.
export const IMAGE_H = 28;
export const IMAGE_W = 28;


const IMAGE_SIZE = IMAGE_H * IMAGE_W;                                   // Разрешение картинок.
const NUM_CLASSES = 10;                                                 // Всего классов картинок (десять чисел)
const NUM_DATASET_ELEMENTS = 65000;                                     // Всего картинок в датасете
const NUM_TRAIN_ELEMENTS = 55000;                                       // Число картинок в обучающих данных
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;    // Число картинок в контрольном сете
const MNIST_IMAGES_SPRITE_PATH = './datasets/mnist_images.png';         // Путь к спрайту с картинками 
const MNIST_LABELS_PATH ='./datasets/mnist_labels_uint8';               // Файл разметки



export class MnistData {
    constructor() {}
    
    // Функция загрузки данных
    async load(){
        // Чтобы получить числовые данные о пикселях на картинке, нужно создать canvas. 
        const img = new Image();
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        /* const htmlElement = document.querySelector('#canvas');
        htmlElement.append(canvas); */

        // Как загрузится изображение-датасет, произойдет его обработка.
        await new Promise((resolve, reject) => {                                                                            // Нужен промис, чтобы подержать load() в ожидании, пока данные не загрузятсья.
            img.addEventListener('load', async () => {
                const chunkSize = 5000;                                                                                     // Указываем размер куска изображения-датасета (загружать нужно кусками за несколько итераций, потому что память браузера ограничена и он упадет, если скормить ему большую пачку данных)
                canvas.width = img.width;
                canvas.height = chunkSize;
                const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);                          // Поскольку нельзя создать огромный массив с 65000 картинками, ибо браузер упадет, а типизированный массив не получится дополнять поитерабельно кусками, ведь он фиксированный, поэтому создадим фиксированный буфер, в который будем поитерабельно собирать данные. 4 - означает, что мы собираем данные формата 32-бит, то есть данные 4-канальной (RGBA) картинки, таковую отдает нам canvas.  [[0,0,0,0],[0,0,0,0],[r,g,b,a]...] - представление картики в 32 bit

                for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {                                                // Итерация для сборки массива данных из кусков по 5000 строк (картинок).
                    const datasetBytesView = new Float32Array(                                                              // datasetBytesView служит замыканием на внутренний цикл. При заполнении datasetBytesView, datasetBytesBuffer поитерабельно накопит все данные, поскольку является его основанием, то есть является его буфером.
                        datasetBytesBuffer,
                        i * IMAGE_SIZE * chunkSize * 4,                                                                     // Эти данные указывают откуда заполнять (смещение - byteOffset),
                        IMAGE_SIZE * chunkSize                                                                              // и сколько (какой длины - length)
                    );
                    ctx.drawImage(img, 0, i * chunkSize, img.naturalWidth, chunkSize, 0, 0, img.naturalWidth, chunkSize);   // Рисуем в canvas каждый кусок изображения-датасета.
                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);                                  // Забираем данные в виде Uint8ClampedArray из нарисованного изображения. В нем содержаться числовые данные о яркости пикселов от 0 до 255. 
                    for (var j = 0; j < imageData.data.length / 4; j++) {                                                   // Проходим по типизированному Uint8ClampedArray и записываем каждый четвертый элемент (байт) в новый массив с приведением данных к формату от 0 до 1 с плавающей точкой. 4 - означает, что для 8 битной картинки в 32-битном представлении нужно обращать внимание только на каждый 4 байт, т.к. canvas содержит 32-битную картинку, данные которой представлены в виде четырех каналов - RGBA от 0 до 255, а не одного, как в нашем датасете.
                        datasetBytesView[j] = imageData.data[j * 4] / 255;                                                  // Пишем в datasetBytesView, чтобы заполнить его буфер datasetBytesBuffer.
                    }
                }
                this.datasetImages = new Float32Array(datasetBytesBuffer);                                                  // Еще раз обратимся к буферу, который собрал в себя все данные и преобразуем в типизированный массив Float32Array.
                // После того как данные изображения-датасета загружены и обработаны, загружаем данные разметки.
                const labelsResponse = await fetch(MNIST_LABELS_PATH);
                this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

                // Затем подготавливаем (нарезаем) тренировочные и контрольные наборы данных.
                this.trainImages = this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
                this.testImages  = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
                this.trainLabels = this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
                this.testLabels  = this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
                resolve();                                                                                                  // Когда все загруженные данные записались в св-ва MnistData, сообщаем промису, что он завершен, после чего функция easyLoad завершится.
            });

            // Указываем путь изображения-датасета, который запустит событие load.
            img.src = MNIST_IMAGES_SPRITE_PATH;
        });
    }


    // Функция загрузки данных c помощью функций Tensorflow.
    async loadWithTfAPI(){
        const img = new Image();
     
        const data = {
            train: {
                xs: null,
                labels: null,
            },
            test: {
                xs: null,
                labels: null,
            }
        };

        await new Promise((resolve, reject) => {                            // Нужно подождать пока картинка загрузится, обработать ее данные и только тогда промис отдаст разрешение на переход к return.
            img.addEventListener('load', async () => {
                let x = tf.browser.fromPixels(img, 1).asType('float32');        // fromPixels(медиаэлемент, [каналов])
                x = x.div(255);                                                 // необходимо данные нормализовать к диапазону от 0 до 1, если данные в тензоре входят в другой диапазон, скажем 0–255 (стандартный набор цветов пикселя на один канал), поскольку сеть учится на нормализованных данных.
                data.train.xs = x.slice([0, 0, 0], [55000, 784, 1]).reshape([55000, 28, 28, 1]);            // Отрезаем кусок тензора для тренировочных данных и меняем его форму (мерность).
                data.test.xs  = x.slice([55000, 0, 0], [10000, 784, 1]).reshape([10000, 28, 28, 1]);

                // После того как данные изображения-датасета загружены и обработаны, загружаем данные разметки.
                const labelsResponse = await fetch(MNIST_LABELS_PATH);
                this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());
                let labelsTensor = tf.tensor2d(this.datasetLabels, [this.datasetLabels.length / NUM_CLASSES, NUM_CLASSES]);
                data.train.labels = labelsTensor.slice([0, 0], [55000, NUM_CLASSES]);
                data.test.labels = labelsTensor.slice([55000, 0], [10000, NUM_CLASSES]);
                resolve();
            });
            img.src = MNIST_IMAGES_SPRITE_PATH;
        });
        return data;
    }



    // Функция подготовки обучающих тензоров
    getTrainData() {
        const xs = tf.tensor4d(this.trainImages, [this.trainImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]);          // features [размер батча, высота, ширина, цветовых каналов]
        const labels = tf.tensor2d(this.trainLabels, [this.trainLabels.length / NUM_CLASSES, NUM_CLASSES]);             // targets  [размер батча, кол-во классов]
        return {xs, labels};
    }

    // Функция подготовки тестовых тензоров
    getTestData(numExamples){
        let xs = tf.tensor4d(this.testImages, [this.testImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]);
        let labels = tf.tensor2d(this.testLabels, [this.testLabels.length / NUM_CLASSES, NUM_CLASSES]);
        // Если указано иное количество примеров данных (размер батча). Используется для визуализации картинок при проверке модели.
        if (numExamples != null) {
            xs = xs.slice([0, 0, 0, 0], [numExamples, IMAGE_H, IMAGE_W, 1]);
            labels = labels.slice([0, 0], [numExamples, NUM_CLASSES]);
        }
        return {xs, labels};
    }



    // Отдельная функция подготовки тестовых тензоров длиной numExamples для визуализации картинок при проверке модели.
    getTestDataForVisual(testTensor, numExamples){
        let xs = testTensor.xs.slice([0, 0, 0, 0], [numExamples, 28, 28, 1]);
        let labels = testTensor.labels.slice([0, 0], [numExamples, NUM_CLASSES]);
        return {xs, labels};
    }

}