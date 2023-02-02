const tf = require('@tensorflow/tfjs');

import spectrogramCreator from './audioToFreqData.js';
import * as ui from './ui';


const datasetPath = 'dataset/mini_speech_commands/';
const labelNames = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes'];

class SpeechData {
    // Имена файлов датасета распледеленные по классам.
    fileNamesByClasses = {};
    // Флаг готовности данны, то есть когда все данные загружены и обработаны.
    spectroDataReady = false;
    // Количество примеров контрольных данных в процентах от общего числа данных. 
    testDataElementsPercent = 15;
    // Объект для определения минимальной и максимальной длины аудио в "кадрах".
    spectroDataLength = {min: 10000, max: 0};
    // Ограничитель на кол-во загружаемых данных из датасета. False - без ограничений. N - целое число примеров от каждого класса. Нужно, так как браузер не в состоянии "переварить" все данные. 
    limit = 10;



    // Количественная информация о датасете (формируется в процесса загрузки данных).
    dataInfo = {
        classes: [],                 // Классы датасеты (формируется в процессе загрузки данных).
        elemsNum: 0,                // Общее количество эелементов в датасете (всего).
        classesNum: 0,               // Всего классов в датасете
        elemsPerClass: 0,           // Элементов в классе.
        spectroSize: {x: 0, y: 0},  // Размеры спектрограмм, где X - кол-во кадров (отрезков времени) аудио; Y - кол-во частотных областей.
        loadedElements: 0,          // Кол-во загруженных элементов датасета (они могут быть загружены не все, в зависимости от свойства limit или от кол-ва элементов в JSON-файле подготовленных данных.)
    };

    // Массивы данных загруженного датасета.
    dataset = {
        spectro: {                  // Частотные данные полученные из файлов датасета, которые можно представить в виде спектрограмм, а также данные о классах. 
            data: [],
            labels: []
        },
        shuffled: {                 // Частотные данные датасета перемешанные в случайном порядке.
            data: [],
            labels: []
        },
        flat: {                     // Сплющенные в одномерные массивы данные датасета из data.shuffled.
            data: [],
            labels: []
        }
    }


    /////////////////////////////////////////////////////////////////
    ///// Оберточный метод загрузки аудиоданных и их обработки. /////
    /////////////////////////////////////////////////////////////////
    async preparingData(){
        await this.getDatasetFileNamesByClases();
        spectrogramCreator.fftSize = ui.preparingDataOptions.fftSize;
        spectrogramCreator.bufferSize = ui.preparingDataOptions.bufferSize;
        this.getSpectroData();
        // Ожидаем, пока последний рекурсивно вызванный метод getSpectroData изменит флаг spectroDataReady, что будет означать, что все файлы загружены и обработаны.
        await new Promise((resolve, reject) => {
            let id = setInterval(() => {
                if(this.spectroDataReady){
                    clearInterval(id);
                    resolve();
                }
            }, 500);
        });
        // Стандартизация всех данных спектрограмм к одному размеру.
        this.standardizationOfLength();
        // Удаление высоких частот из спектрограм (отрезание верхней половины), если отмечено. 
        if(ui.preparingDataOptions.deleteHF) this.cutting();
        // Перемешивание данных
        this.shufflerData();
        // Уплощение многомерных массивов данных в однномерный для последующего удобного преобразоования в тензоры.
        this.flattenData();
        ui.flattenDataIsReady();
    }


    //////////////////////////////////////////////////////////////////
    ///// Метод загрузки готовых "плоских" данных из JSON-файла. /////
    //////////////////////////////////////////////////////////////////
    async getReadyDataFromJSONFile(){
        const response =  await fetch('/dataset/msc_ready_data.json', { method:'GET' });
        if(!response.ok){
            this.loadingStatus = `Запрошенный файл не дал ответа! Неверно указан адрес или файл не существует! ('/dataset/msc_ready_data.json)`;
            console.warn(this.loadingStatus);
            return;
        }
        const mscData = await response.json();
        this.dataInfo = mscData.dataInfo;
        this.dataset.flat = mscData.dataset;
        delete this.dataset.spectro;
        delete this.dataset.shuffled;
        ui.flattenDataIsReady(true);
    }



    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///// Функция забирает имена файлов из заранее подготовленного списка файлов и помещает их в классифицированный объект с массивами имен этих фалов соответсвенно классам. /////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    async getDatasetFileNamesByClases(){
        const response =  await fetch('/dataset/msc_dataset.json', { method:'GET' });
        if(!response.ok){
            this.loadingStatus = `Запрошенный файл не дал ответа! Неверно указан адрес или файл не существует! (/dataset/msc_dataset.json)`;
            console.warn(this.loadingStatus);
            return;
        }
        this.fileNamesByClasses = await response.json();
        // Получаем размеры данных и имена классов.
        for (const cls in this.fileNamesByClasses) {
            this.dataInfo.classes.push(cls);
            this.dataInfo.elemsNum += this.fileNamesByClasses[cls].length;
            this.dataInfo.elemsPerClass = this.fileNamesByClasses[cls].length;
            this.dataInfo.classesNum++;
        }
    }



    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///// Метод получает порции данных рекукрсивно: частотные данные в виде массива (виртуальную спектрограмму) одного файла от каждого класса. /////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    async getSpectroData(elemsNum = 0){
        // Загрузка и получение массива частотных данных в виде Uint8Array из файлов.
        const d = {};
        let files = 0;
        for (const className in this.fileNamesByClasses) {
            const sd = await spectrogramCreator.getFreqDataFromFile('/dataset/'+className+'/'+this.fileNamesByClasses[className][elemsNum]);
            this.dataset.spectro.data.push(sd);
            this.dataset.spectro.labels.push(this.dataInfo.classes.findIndex(el => el == className));
            files++;
            // Поскольку файлы могут быть разной длины, найдем самый длинный, чтобы затем все остальные "нарастить" до такой же длины.
            this.spectroDataLength.max = sd.length > this.spectroDataLength.max ? sd.length : this.spectroDataLength.max;
            this.spectroDataLength.min = sd.length < this.spectroDataLength.min ? sd.length : this.spectroDataLength.min;
        }
        // Ждем пока все файлы от каждого класса обработаются с помощью отслеживающего интервала.
        await new Promise((resolve, reject) => {
            let id = setInterval(() => {
                if(files == this.dataInfo.classesNum){
                    clearInterval(id);
                    resolve();
                }
            }, 100);
        });
        // После того как первая стопка файлов (по каждому файлу от класса) обработана, устанавливаем счетчик +1 и запускаем загрузку следующей и так пока весь датасет не будет обработан или не будет достигнут указанный предел.
        elemsNum++;
        if(elemsNum < (this.limit || this.dataInfo.elemsPerClass)){
            this.getSpectroData(elemsNum);
            ui.spectroDataProgress(100 * (elemsNum * this.dataInfo.classesNum) / this.dataInfo.elemsNum);
        }else{
            // Установим флаг завершения обработки данных.
            this.spectroDataReady = true;
            // Запишем стандартизированные размеры спектрограм для дальнейшего использования при создании тензоров. 
            this.dataInfo.spectroSize.x = this.spectroDataLength.max;
            this.dataInfo.spectroSize.y = spectrogramCreator.fftSize / 2;
            this.dataInfo.loadedElements = this.dataset.spectro.data.length;
        }
    }



    /////////////////////////////////////////////////////////////////////////////////////
    ///// Функция наполнитель, создает голову и хвост необходимой переданной длины. /////
    /////////////////////////////////////////////////////////////////////////////////////
    filling(head, tail){
        const headArr = [];
        for(let i = 0; i < head; i++){
            headArr.push(new Uint8Array(this.dataInfo.spectroSize.y));
        }
        const tailArr = [];
        for(let i = 0; i < tail; i++){
            tailArr.push(new Uint8Array(this.dataInfo.spectroSize.y));
        }
        return [headArr, tailArr];
    };


    ////////////////////////////////////////////////////////////////////////////////////////////////////
    ///// Функция обрезания в данных верхних частот для уменьшения размера спектрогармм по высоте. /////
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    cutting(){
        this.dataInfo.spectroSize.y = this.dataInfo.spectroSize.y / 2;
        this.dataset.spectro.data.forEach((sound, key) => {
            sound.forEach((frame, kf) => {
                const newData = [];
                frame.forEach((el, k) => {
                    if(k < this.dataInfo.spectroSize.y) newData.push(el);
                });
                this.dataset.spectro.data[key][kf] = newData;
            });
        });
    }



    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///// Метод стандартизирует ширину спектрограммы (длительность звука) под самый широкий экземпляр, добавляя пустые данные по краям более коротким экземплярам. /////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    standardizationOfLength(){
        let maximum = 0;
        let totalMax = 0;
        let totalH = 0;
        // Пробегаемся по массиву всех данных и если там есть массивы меньше максимальной длины, то дополняем их пустыми данными по краям, предвариртельно посчитав, сколько элементов нужно добавить.
        this.dataset.spectro.data.forEach((element, key) => {
            if(element.length < this.spectroDataLength.max){
                const half = (this.spectroDataLength.max - element.length)  / 2;
                const edges = (half % 2 === 0) ? this.filling(half, half) : this.filling(Math.ceil(half), Math.floor(half))
                this.dataset.spectro.data[key] = edges[0].concat(element, edges[1]);
            }
        });
    }



    /////////////////////////////////////
    ///// Метод перемешивает данные /////
    /////////////////////////////////////
    shufflerData(){
        // Перемешивание индексов случайным образом.
        const indexes = [];
        for (let i = 0; i < this.dataset.spectro.data.length; ++i) {
            indexes.push(i);
        }
        tf.util.shuffle(indexes);
        // Сборка новых массивов с данными и кассами на основе перемешанных индексов.
        for (let i = 0; i < this.dataset.spectro.data.length; ++i) {
            this.dataset.shuffled.data.push(this.dataset.spectro.data[indexes[i]]);
            this.dataset.shuffled.labels.push(this.dataset.spectro.labels[indexes[i]]);
        }

        delete this.dataset.spectro;
    }


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///// Метод сплющивает разнотипизированный многомерный массив в обычный для удобного преобразования в тензоры. /////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    flattenData(){
        const arr = [1,2,3,4,5]
        if(Array.isArray(this.dataset.shuffled.data[0][0])){                // Если частотные данные не являются типизированным массивом, а обычный массив (когда была произведена обрезка).
            this.dataset.flat.data = this.dataset.shuffled.data.flat(2);
        }else{                                                              // Но если типизированный, то для него не сработает метод flat и требуется дополнительный проход по нему.
            this.dataset.shuffled.data.flat(2).forEach(el => {
                el.forEach(value => this.dataset.flat.data.push(value));
            });
        }
        this.dataset.flat.labels = this.dataset.shuffled.labels;
        delete this.dataset.shuffled;
    }


    ///////////////////////////////////////////////////////////////////////////
    ///// Метод сохранения обработанных данных из аудиофайлов в JSON-файл /////
    ///////////////////////////////////////////////////////////////////////////
    // Создание файла данных, готовых для преобразования их в тензоры.
    // Наличие данного файла позволяет избежать длительного процесса преобразования аудиоданных из аудиофайлов каждый раз,
    // когда требуется перезагрузить страницу. Файл предоставляет уже преобразованные данные.
    safeFlatDataInFile(){
        const dataJSON = JSON.stringify({
            dataInfo: this.dataInfo,
            dataset: this.dataset.flat
        });
        function utoa(str) {                // ucs-2 string to base64 encoded ascii
            return window.btoa(unescape(encodeURIComponent(str)));
        }
        const type = 'data:application/octet-stream;base64, ';
        const base = utoa(dataJSON);
        return type + base;
    }



    createTensors(){
        return tf.tidy(() => {
            // Приведение 8 битных данных о частоте в диаппазон от 0 до 1 с плавающей точкой. Это необходимо для SGD.
            this.dataset.flat.data.forEach((el, key) => {
                this.dataset.flat.data[key] = el / 255; 
            });


            // Теперь можно создать тензор данных из сплющенного массива, он будет формы [8000, 128, 187, 1] для полного объема датасета. 1 - это кол-во каналов.
            const commonDataTensor = tf.tensor4d(
                this.dataset.flat.data,
                [
                    this.dataset.flat.data.length / (this.dataInfo.spectroSize.y * this.dataInfo.spectroSize.x),
                    this.dataInfo.spectroSize.y,
                    this.dataInfo.spectroSize.x,
                    1
                ]
            );
            // Тензор классов создается приведением к одномерному тензору, а затем его унитарным кодированием, что в итоге вернет двумерный тензор формы [8000, 8]. 
            const commonLabelsTensor = tf.oneHot(tf.tensor1d(this.dataset.flat.labels).toInt(), this.dataInfo.classesNum);
            const testNum = Math.round(this.dataInfo.loadedElements * this.testDataElementsPercent / 100);
            // Делим общий тензор данных на тренировочный и контрольный.
            const trainData = commonDataTensor.slice([0, 0, 0, 0], [this.dataInfo.loadedElements - testNum, this.dataInfo.spectroSize.y, this.dataInfo.spectroSize.x, 1]) 
            const testData = commonDataTensor.slice([this.dataInfo.loadedElements - testNum, 0, 0, 0], [testNum, this.dataInfo.spectroSize.y, this.dataInfo.spectroSize.x, 1])
            // Так же делим тензор классов на тренировочный и контрольный.
            const trainLabels = commonLabelsTensor.slice([0, 0], [this.dataInfo.loadedElements - testNum, this.dataInfo.classesNum]);
            const testLabels = commonLabelsTensor.slice([this.dataInfo.loadedElements - testNum, 0], [testNum, this.dataInfo.classesNum]);

            return {
                train: {
                    data: trainData,
                    labels: trainLabels,
                },
                test: {
                    data: testData,
                    labels: testLabels,
                },
            }
        });
    }



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///// Отдельная функция подготовки тестовых тензоров длиной numExamples для визуализации эффективности обучения при проверке модели. /////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    getTestDataForVisual(testTensor, numExamples){
        let dataForVisual = testTensor.data.slice([0, 0, 0, 0], [numExamples, this.dataInfo.spectroSize.y, this.dataInfo.spectroSize.x, 1]);
        let labelsForVisual = testTensor.labels.slice([0, 0], [numExamples, this.dataInfo.classesNum]);
        return {dataForVisual, labelsForVisual};
    }

}



export const speechData = new SpeechData();