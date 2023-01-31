const tf = require('@tensorflow/tfjs');
import './style.css';
import { speechData } from './data';
import * as ui from './ui';


const createHistogramButton = document.getElementById('create-spectrogram');
const loadReadyDataButton = document.getElementById('load-json-data');
let tensors = {};       // Тензоры для обучения.
let examples = {};      // Часть примеров-тензоровы для проверки модели.


createHistogramButton.addEventListener('click', async (ev) => {
    ui.spectroDataProgress(0, ev.target);
    await speechData.preparingData();
});



loadReadyDataButton.addEventListener('click', async (ev) => {
    await speechData.getReadyDataFromJSONFile();
});



function createModel(type){
    if (type == 'convnet'){       return createConvModel();}
    else if (type == 'densenet'){ return createDenseModel();}
    else {                        throw new Error(`Invalid model type: ${modelType}`);}
};



function createDenseModel() {
    const model = tf.sequential();
    model.add(tf.layers.flatten({inputShape: [speechData.dataInfo.spectroSize.y, speechData.dataInfo.spectroSize.x, 1]}));
    model.add(tf.layers.dense({units: 400, activation: 'relu'}));
    //tf.layers.dropout({rate: 0.25}), // В данной модели с данными низкого разрешения слои дропаута почти никак не влияют на эффективность обучения. При большем разрешении спектрограмм наблюдается незначительный эффект снижения переобучения.
    model.add(tf.layers.dense({units: 42, activation: 'relu'}));
    //tf.layers.dropout({rate: 0.5}),    
    model.add(tf.layers.dense({units: speechData.dataInfo.classesNum, activation: 'softmax'}));
    model.summary();
    return model;
}



export function createConvModel(){
/*     const model = tf.sequential({
        layers: [
            tf.layers.conv2d({                          // На входе [28, 28, 1], => на выходе дает тензор [26, 26, 16].  26 - кол-во перемещений ядра (окна) размером 3 по исходной картинке по высоте или по ширине.
                inputShape: [speechData.dataInfo.spectroSize.y, speechData.dataInfo.spectroSize.x, 1],          // высота, ширина, глубина (1 канал).
                kernelSize: [2, 8],                         // размер ядра 2 на 8 пикселя. Окно 2 на 8 будет шагать по основному изображению по всем возможным положениям и возвращать эти миникартинки (тензоры) фильтру, который затем разобьет его на 8 срезов и выполнит склярное произведение H*W*C для каждого среза, вернув новый 3d-тензор вида: [кол-во шагов окна по ширине входного изображения, высоте, глубину 8], где каждое значение будет хранить произведение срезов. 
                filters: 8,                                 // количество фильтров в свертке, каналов выходного сигнала или глубина сверточного ядра - по сути кол-во усваиваемых признаков входного изображения, например: прямолинейных границ, углов цвета и т.д.
                activation: 'relu'
            }),
            tf.layers.maxPooling2d({                    // Операция субдискретизации с выбором максимального значения из "окна" для каждого измерения глубины. Таким образом, если входной тензор [26, 26, 16], то выходной получится [13, 13, 16]. Слой maxPooling2d также уменьшает чувствительность к положению распознаваемого символа на холсте. 
                poolSize: 2,                                // размер "окна" или поля, подобное ядру в conv2d. 2 на 2.
                strides: 2                                  // шаг 2 означает, что "окно" шагает по две "клетки"
            }),
            tf.layers.conv2d({                          // на входе [13, 13, 16], на выходе [11, 11, 32]
                kernelSize: [2, 4],
                filters: 32,
                activation: 'relu'
            }),
            tf.layers.maxPooling2d({
                poolSize: 2,
                strides: 2
            }),
            tf.layers.flatten(),                            // Слой схлопывания «сплющивает» многомерный тензор в одномерный с сохранением общего числа элементов. Это подготовка для плотных слоев.
            tf.layers.dropout({rate: 0.25}),
            tf.layers.dense({
                units: 2000,
                activation: 'relu'
            }),
            tf.layers.dropout({rate: 0.5}),
            tf.layers.dense({
                units: speechData.dataInfo.classesNum,
                activation: 'softmax'
            })

        ]
    }); */

    const model = tf.sequential({
        layers: [
            tf.layers.conv2d({                          // На входе [28, 28, 1], => на выходе дает тензор [26, 26, 16].  26 - кол-во перемещений ядра (окна) размером 3 по исходной картинке по высоте или по ширине.
                inputShape: [speechData.dataInfo.spectroSize.y, speechData.dataInfo.spectroSize.x, 1],          // высота, ширина, глубина (1 канал).
                kernelSize: 3,                              // размер ядра 3 на 3 пикселя. Окно 3 на 3 будет шагать по основному изображению по всем возможным положениям и возвращать эти миникартинки (тензоры) фильтру, который затем разобьет его на 16 срезов и выполнит склярное произведение H*W*C для каждого среза, вернув новый 3d-тензор вида: [кол-во шагов окна по ширине входного изображения, высоте, глубину 16], где каждое значение будет хранить произведение срезов. 
                filters: 16,                                // количество фильтров в свертке, каналов выходного сигнала или глубина сверточного ядра - по сути кол-во усваиваемых признаков входного изображения, например: прямолинейных границ, углов цвета и т.д.
                activation: 'relu'
            }),
            tf.layers.maxPooling2d({                    // Операция субдискретизации с выбором максимального значения из "окна" для каждого измерения глубины. Таким образом, если входной тензор [26, 26, 16], то выходной получится [13, 13, 16]. Слой maxPooling2d также уменьшает чувствительность к положению распознаваемого символа на холсте. 
                poolSize: 2,                                // размер "окна" или поля, подобное ядру в conv2d. 2 на 2.
                strides: 2                                  // шаг 2 означает, что "окно" шагает по две "клетки"
            }),
            tf.layers.conv2d({                          // на входе [13, 13, 16], на выходе [11, 11, 32]
                kernelSize: 3,
                filters: 32,
                activation: 'relu'
            }),
            tf.layers.maxPooling2d({
                poolSize: 2,
                strides: 2
            }),
            tf.layers.flatten(),                            // Слой схлопывания «сплющивает» многомерный тензор в одномерный с сохранением общего числа элементов. Это подготовка для плотных слоев.
            tf.layers.dense({
                units: 64,
                activation: 'relu'
            }),
            tf.layers.dense({
                units: 8,
                activation: 'softmax'
            })

        ]
    });

    model.summary();
    return model;
}



async function train(model, epochs) {
    model.compile({
        optimizer: 'rmsprop', // tf.train.sgd(0.05), // 'rmsprop'  - показал себя эффективней, 
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    const batchSize = 250;                  // Преимущество большого размера батчей заключается в более согласованном и менее подверженном изменениям градиентном обновлении весов, но требует больше оперативки.
    const validationSplit = 0.05;
    const totalNumBatches = Math.ceil(tensors.train.data.shape[0] * (1 - validationSplit) / batchSize) * epochs;      // Math.ceil(55000  *  (1 - 0.15)  /  320) * 3 = 441  всего батчей.
    ui.showModelParams({
        batchSize,
        totalNumBatches,
        epochs,
        shape: tensors.train.data.shape[1]+' × '+tensors.train.data.shape[2],
        trainData: tensors.train.data.shape[0],
        testData: tensors.test.data.shape[0]

    });

    let   trainBatchCount = 0;
    let   currentEpoch = 0;
    let   valAcc;

    await model.fit(tensors.train.data, tensors.train.labels, {
        batchSize,
        validationSplit,            // Проверочные данные - последние 15% от обучающих.
        epochs: epochs,
        callbacks: {
            onBatchEnd: (batch, logs) => {
                trainBatchCount++;
                ui.logStatus({mess: `Батч ${trainBatchCount} из ${totalNumBatches} | ${(trainBatchCount / totalNumBatches * 100).toFixed(1)}%  завершено.`, epoch: currentEpoch}, 'train');
                ui.plotLoss({
                    currentEpoch: currentEpoch,
                    totalEpochs: epochs,
                    currentBatch: trainBatchCount,
                    batchSize: totalNumBatches,
                    loss: logs.loss
                }, 'batchEnd');
                ui.plotAccuracy({
                    currentEpoch: currentEpoch,
                    totalEpochs: epochs,
                    currentBatch: trainBatchCount,
                    batchSize: totalNumBatches,
                    acc: logs.acc
                }, 'batchEnd')

                if(trainBatchCount % 10 === 0){             // Отображать результативность предсказаний модели на картинках через каждые 10 батчей.
                    //showPredictions(model);
                }
            },
            onEpochEnd: (epoch, logs) => {
                ui.plotLoss({
                    currentEpoch: currentEpoch,
                    totalEpochs: epochs,
                    currentBatch: trainBatchCount,
                    batchSize: totalNumBatches,
                    val_loss: logs.val_loss
                }, 'epochEnd');

                ui.plotAccuracy({
                    currentEpoch: currentEpoch,
                    totalEpochs: epochs,
                    currentBatch: trainBatchCount,
                    batchSize: totalNumBatches,
                    val_acc: logs.val_acc
                }, 'epochEnd');

                showPredictions(model);

                currentEpoch++;
                valAcc = logs.val_acc;
            },
        }
    });
}



async function showPredictions(model) {
    const testExamples = tensors.test.data.shape[0] > 50 ? 50 : tensors.test.data.shape[0];
    examples = speechData.getTestDataForVisual(tensors.test, testExamples);

    //console.log(tensors.test)

//    console.log(examples)
    /* tf.argMax() возвращает индексы максимальных значений в тензоре вдоль определенной оси. Задачи категориальной классификации, подобные этой, часто представляют классы в виде одномерных векторов.
    Одномерные векторы - это одномерные векторы с одним элементом для каждого выходного класса. Все значения в векторе равны 0 за исключением одного, который имеет значение 1 (например [0, 0, 0, 1, 0]).
    Результатом model.predict() будет распределение вероятностей, поэтому мы используем argMax, чтобы получить индекс векторного элемента, который имеет наибольшую вероятность. Это наш прогноз.
    (например, argmax([0.07, 0.1, 0.03, 0.75, 0.05]) == 3) функция dataSync() синхронно загружает значения tf.tensor из графического процессора, чтобы мы могли использовать их в нашем обычном
    коде CPU JavaScript (для неблокирующей версии этой функции используйте data()). */

    tf.tidy(() => {     // Код, заключенный в обратный вызов функции tf.tidy(), будет иметь свои тензоры, освобожденные из памяти графического процессора после выполнения без необходимости вызывать dispose(). Обратный вызов tf.tidy выполняется синхронно. tf.tidy() предотвращает утечки памяти WebGL.
        const output = model.predict(examples.dataForVisual);                           // Модель возвращает классы в виде тензора.
        const axis = 1;
        const labels = Array.from(examples.labelsForVisual.argMax(axis).dataSync());    // Метки (классы) размеченных данных (массив чисел от 0 до 10)
        const predictions = Array.from(output.argMax(axis).dataSync());                 // Предсказания (массив чисел от 0 до 10)
        ui.showTestResults(examples, predictions, labels);                              // Визуально сверяем на примерах данных предсказания с метками классов.
    });
}



export async function runButtonCallback(params){
    ui.logStatus({mess: `Создание ${params.modelType}-модели...`}); 
    const model = {model: createModel(params.modelType)} ;
    ui.logStatus({mess: `Обучение модели:`});
    await train(model.model, params.epochs);
    //showPredictions(model.model);
}



export function createTensors(){
    tensors = speechData.createTensors();
    ui.tensorsComplete();
    console.log(tensors.train);
    console.log(tensors.test)
}