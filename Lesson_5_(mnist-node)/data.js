const tf = require('@tensorflow/tfjs');
//const assert = require('assert');
const fs = require('fs');
const assert = require('assert');
//const https = require('https');
const util = require('util');
const zlib = require('zlib');



// Данные о пути к файлам картинок и разметки.
const BASE_URL = './data/';
const TRAIN_IMAGES_FILE = 'train-images-idx3-ubyte.gz';
const TRAIN_LABELS_FILE = 'train-labels-idx1-ubyte.gz';
const TEST_IMAGES_FILE = 't10k-images-idx3-ubyte.gz';
const TEST_LABELS_FILE = 't10k-labels-idx1-ubyte.gz';

// Данные о базе данных (файлах) тренировочных и контрольных изображений.
const IMAGE_HEIGHT = 28;
const IMAGE_WIDTH = 28;
const IMAGE_FLAT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;
const IMAGE_HEADER_MAGIC_NUM = 2051;        // Магические числа — это первые несколько байтов (32 бит или 4 байта) файла, которые уникальны для определенного типа файла. Эти уникальные биты называются магическими числами, которые также иногда называют сигнатурой файла. Эти байты могут использоваться системой для «различения и распознавания разных файлов» без расширения файла. Это число представляет собой беззнаковое целое с порядком прочтения high-endian и представляет собой число 2051 в десятичной системе счисления. 
const IMAGE_HEADER_BYTES = 16;              // Число байтов в заголовке. Т.е. с 16 байта идут уже данные о пикселях изображений. (см. подробности ниже.)
/*
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel

Т.е. грубо говоря, можно проигнорировать заголовок, отмотать сразу на 16-й байт и прочитать 784 байта первой картинки. Каждый байт это яркость очередного серого (одноканального) пикселя: 0 белый, 255 черный.
*/

// Данные о базе данных (файлах) тренировочных и контрольных данных разметки.
const LABEL_HEADER_MAGIC_NUM = 2049;        // Магические число, здесь оно 2049. 
const LABEL_HEADER_BYTES = 8;               // Число байтов в заголовке. Т.е. с 8 байта идут данные разметки.
const LABEL_RECORD_BYTE = 1;                // Сколько байт в элементе будет присутсвовать в элементе массива меток.
const LABEL_FLAT_SIZE = 10;                 // Размер плоскости меток (вариантов меток).




/////////////////////////////////////////////////////////////////////
///// Функция сбора байтов в массив для последующего сравнения. /////
/////////////////////////////////////////////////////////////////////
function loadHeaderValues(buffer, headerLength) {
    const headerValues = [];
    for (let i = 0; i < headerLength / 4; i++) {
        headerValues[i] = buffer.readUInt32BE(i * 4);     // Данные заголовка хранятся по порядку (он же big-endian)
    }
    return headerValues;
}



////////////////////////////////////////////////////////
///// Функция загрузки файла и записи его в буфер. /////
////////////////////////////////////////////////////////
async function getBufferFromFile(fileName){
    const filename = BASE_URL + fileName;
    return new Promise((resolve, reject) => fs.readFile(filename, {/*encoding: 'utf-8'*/}, (err , data) => {
        if(err) return reject(err.message);
        // Распаковка .gz. Если файл не запакован, то можно сразу возвращать buffer.
        zlib.unzip(data, (err, buffer) => {
            resolve(buffer);
        }); 
    }))
}



//////////////////////////////////////////////////
///// Функция подготовки данных изображений. /////
//////////////////////////////////////////////////
async function loadImagesFromFile(fileName) {
    let buffer = await getBufferFromFile(fileName);
    const headerBytes = IMAGE_HEADER_BYTES;
    const recordBytes = IMAGE_FLAT_SIZE;

    //const headerValues = loadHeaderValues(buffer, headerBytes);
    //assert.equal(headerValues[0], IMAGE_HEADER_MAGIC_NUM);              // assert.equal -  сравнивает значения и если два значения не равны, выдается ошибка и программа завершается.
    //assert.equal(headerValues[2], IMAGE_HEIGHT);
    //assert.equal(headerValues[3], IMAGE_WIDTH);

    const images = [];
    let byte = headerBytes;
    while (byte < buffer.byteLength) {
        const array = new Float32Array(recordBytes);        // Создать массив Float32 с 784 элементами. 32bit - потому что изображения у нас предполагаются именно в таком формате.
        for (let i = 0; i < recordBytes; i++) {
            array[i] = buffer.readUInt8(byte++) / 255;      // Записываем байт из буфера со смещением предварительно нормализовав его, приведя к float значению от 0 до 1. 
        }
        images.push(array);
    }
    return images;
}


///////////////////////////////////////////////
///// Функция подготовки данных разметки. /////
///////////////////////////////////////////////
async function loadLabelsFromFile(fileName) {
    let buffer = await getBufferFromFile(fileName);
    const headerBytes = LABEL_HEADER_BYTES;

    const headerValues = loadHeaderValues(buffer, headerBytes);
    assert.equal(headerValues[0], LABEL_HEADER_MAGIC_NUM);      // assert.equal -  сравнивает значения и если два значения не равны, выдается ошибка и программа завершается.

    const labels = [];
    let byte = headerBytes;
    while (byte < buffer.byteLength) {
        const array = new Int32Array(LABEL_RECORD_BYTE);    // создать целочисленный массив Int32 с одним единственным значением (с одним байтом).
        for (let i = 0; i < LABEL_RECORD_BYTE; i++) {
            array[i] = buffer.readUInt8(byte++);            // Метод Buffer.readUInt8() используется для чтения 8-битного целого числа без знака из объекта Buffer. (Node.js). Метод принимает один параметра - смещение, которое указывает положение объекта в буфере. Он представляет собой количество байтов, которое необходимо пропустить перед началом чтения. 
        }
        labels.push(array);
    }
    return labels;
}




class MnistDataset {

    constructor() {
        this.dataset = null;
        this.trainSize = 0;
        this.testSize = 0;
        this.trainBatchIndex = 0;
        this.testBatchIndex = 0;
    }


    async loadData() {
        // Получаем массивы данных из файлов баз.
        this.dataset = await Promise.all([
            loadImagesFromFile(TRAIN_IMAGES_FILE),
            loadLabelsFromFile(TRAIN_LABELS_FILE),
            loadImagesFromFile(TEST_IMAGES_FILE),
            loadLabelsFromFile(TEST_LABELS_FILE)
        ]);
        // Указываем длины тренировочных и контрольных массивов изображений.
        this.trainSize = this.dataset[0].length;
        this.testSize = this.dataset[2].length;
    }

    getTrainData() {
        return this.getData_(true);
    }
    
    getTestData() {
        return this.getData_(false);
    }


    getData_(isTrainingData){
        let imagesIndex;
        let labelsIndex;
        if (isTrainingData) {       // В зависимоости тренировочные или контрольные
            imagesIndex = 0;
            labelsIndex = 1;
        } else {
            imagesIndex = 2;
            labelsIndex = 3;
        }

        // Проверяем, соответсвует ли кол-во изображений количеству меток. Если нет, выводим ошибку.
        const size = this.dataset[imagesIndex].length;
        tf.util.assert(this.dataset[labelsIndex].length === size, `Несоответствие размера количества изображений (${size}) ` +  `количеству меток (${this.dataset[labelsIndex].length})`);


        // Создаем только по одному большому массиву для хранения пакета изображений и меток. Раннее у нас был массив с элементами, которые содержали массив с одним значением.
        const imagesShape = [size, IMAGE_HEIGHT, IMAGE_WIDTH, 1];
        const images = new Float32Array(tf.util.sizeFromShape(imagesShape));        // sizeFromShape() используется для определения количества элементов. Элементы аргумента imagesShape перемножаются таким образом получается размер, например, [10000, 28, 28, 1] = 7840000.
        const labels = new Int32Array(tf.util.sizeFromShape([size, 1]));

        // Наполняем массив значениями из массивов датасета, созданных из файлов.
        let imageOffset = 0;
        let labelOffset = 0;
        for (let i = 0; i < size; ++i) {
          images.set(this.dataset[imagesIndex][i], imageOffset);
          labels.set(this.dataset[labelsIndex][i], labelOffset);
          imageOffset += IMAGE_FLAT_SIZE;
          labelOffset += 1;
        }

        // На выходе создаем тензоры указанной формы.
        return {
            images: tf.tensor4d(images, imagesShape),                                   // Четырехмерный тензор формы, например, [10000, 28, 28, 1], то есть 10 тыс. значений по 28 элементов, внутри которых 28 элементов по одному числу.
            labels: tf.oneHot(tf.tensor1d(labels, 'int32'), LABEL_FLAT_SIZE).toFloat()  // Одномерный  oneHot-тензор формы [10000, 10], то есть десять тысяч значений по 10 элементов, где значение 1 расположено в том элементе, индекс которого соответсвует угадываемому числу от 0 до 9, а остальные элементы = 0. Такой формат удобен для классификации, поскольку на выходе нейросеть будет иметь количество нейронов равное вариантам ответа.
            /*
            [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0],        // соответсвует числу 8
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],        // соответсвует числу 2
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],        // соответсвует числу 1
             ...
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]         // соответсвует числу 3
            ]
            */
        };
    }

}

module.exports = new MnistDataset();