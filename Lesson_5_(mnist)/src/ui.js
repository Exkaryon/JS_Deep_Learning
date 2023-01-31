
const tfvis = require('@tensorflow/tfjs-vis');

import {runButtonCallback} from './index.js';


const preparationLog = document.querySelector('#logs .preparation');
const trainLog = document.querySelector('#logs .train');
const startButton = document.querySelector('#run button');
const productivityString = document.querySelector('#productivity');
const imagesElement = document.querySelector('#images');



/////////////////////////////////////////////
///// Поулчение параметров из интерфеса /////
/////////////////////////////////////////////
function getParamsFromUI(){
    return {
        modelType: document.querySelector('.control select').value,
        epochs: +document.querySelector('.control input[name="epochs"]').value,
    }
}


//////////////////////////////////////////////
///// Обработчик кнопки запуска обучения /////
//////////////////////////////////////////////
startButton.addEventListener('click', function(){
    const params = getParamsFromUI();
    runButtonCallback(params);
    this.remove();
});



///////////////////////////////////////////////
///// Вывод логов жизненного цикла модели /////
///////////////////////////////////////////////
export function logStatus(log, type){
    if(type == 'train'){
         trainLog.innerHTML = `
            &nbsp; ${log.mess}<br>
            &nbsp; Эпоха: ${log.epoch + 1} <br>`;
    }else if(type == 'final'){
        trainLog.innerHTML += `<br>${log.mess}`;
    }else{
        preparationLog.innerHTML += `<br>${log.mess}`;
    }
}



////////////////////////////////////
///// Отрисовка графика потерь /////
////////////////////////////////////
const seriesValues = [[], []];                              // Данные обучения по Батчам, данные обучения по эпохам
let firstLossOnEpoch = 0;                                   // Первичное значение потерь в каждой эпохе для первой точки графика потерь на проверочных данных. 
export function plotLoss(logs, progressType){
    if(progressType == 'batchEnd'){
        if(!seriesValues[0].length) firstLossOnEpoch = logs.loss;                       // Если это первая итераация, то запишем в первый элемент - точку для линии по эпохам - равную первому значению первого батча.
        seriesValues[0].push({x: seriesValues[0].length, y: logs.loss});                // Массив точек для линии потерь по батчам.
        seriesValues[1].push({x: seriesValues[1].length, y: firstLossOnEpoch});         // Массив для линии потерь на проверочных данных может отображаться только за эпоху, но linechart не может работать с двумя массивами разной длины, поэтому пока наполним его любыми значениями.
    }else if (progressType == 'epochEnd'){
        const startEpochVal = firstLossOnEpoch;
        const endEpochVal = logs.val_loss;
        let currenVal = startEpochVal;
        const batchSizePerEpoch = logs.batchSize / logs.totalEpochs;
        const step = (startEpochVal - endEpochVal) / batchSizePerEpoch;                                             // У нас есть только две точки за эпоху (начало и конец), но нам нужно наполнить промежуточные элементы, которые созданы за батчи с неактуальным значением, так, чтобы получился ровный отрезок.
        for(let i = logs.currentEpoch * batchSizePerEpoch; i < batchSizePerEpoch * (logs.currentEpoch + 1); i++){   // Пробежимся по массиву и запишем в него новые значения, получился ровный отрезок.
            currenVal -= step;
            seriesValues[1][i].y = currenVal;
        }
        firstLossOnEpoch = currenVal;                                                                               // Укажем новую точку начала отрезка линни для следующей эпохи.
        //console.log('Batch: ', seriesValues[0][seriesValues[0].length - 1])
        //console.log('Epoch: ', logs.val_loss, seriesValues[1][seriesValues[1].length - 1])
    }

    tfvis.render.linechart(
        document.getElementById('plotLoss'),
        {   
            values: seriesValues,
            series: ['train per Batch', 'validation per Epoch']
        },
        {
            xLabel: 'Batch #',
            yLabel: 'Loss',
            width: 450,
            height: 320,
        }
    );
}


//////////////////////////////////////
///// Отрисовка графика точности /////
//////////////////////////////////////
const seriesValuesAcc = [[], []];
let firstAccOnEpoch = 0;
export function plotAccuracy(logs, progressType){
    if(progressType == 'batchEnd'){
        if(!seriesValuesAcc[0].length) firstAccOnEpoch = logs.acc;
        seriesValuesAcc[0].push({x: seriesValuesAcc[0].length, y: logs.acc});
        seriesValuesAcc[1].push({x: seriesValuesAcc[1].length, y: firstAccOnEpoch});
    }else if (progressType == 'epochEnd'){
        const startEpochVal = firstAccOnEpoch;
        const endEpochVal = logs.val_acc;
        let currenVal = startEpochVal;
        const batchSizePerEpoch = logs.batchSize / logs.totalEpochs;
        const step = (startEpochVal - endEpochVal) / batchSizePerEpoch; 
        for(let i = logs.currentEpoch * batchSizePerEpoch; i < batchSizePerEpoch * (logs.currentEpoch + 1); i++){
            currenVal -= step;
            seriesValuesAcc[1][i].y = currenVal;
        }
        firstAccOnEpoch = currenVal;
    }

    tfvis.render.linechart(
        document.getElementById('plotAccuracy'),
        {   
            values: seriesValuesAcc,
            series: ['train per Batch', 'validation per Epoch']
        },
        {
            xLabel: 'Batch #',
            yLabel: 'Accuracy',
            width: 450,
            height: 320,
        }
    );
}





export function draw(image, canvas) {
    const [width, height] = [28, 28];
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(width, height);
    const data = image.dataSync();                          // dataSync() - Синхронно загружает значения из tf.Тензор,кторые затем можно вывести в консоль или, например, преобразовать в массив: array.from(image.dataSync())
    for (let i = 0; i < height * width; ++i) {
        const j = i * 4;
        imageData.data[j + 0] = data[i] * 255;
        imageData.data[j + 1] = data[i] * 255;
        imageData.data[j + 2] = data[i] * 255;
        imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}




export function showTestResults(exampleBatch, predictions, labels) {

    let productivity = 0;
    const testExamples = exampleBatch.xs.shape[0];              // exampleBatch.xs.shape = [100, 28, 28, 1]
    imagesElement.innerHTML = '';

    for (let i = 0; i < testExamples; i++) {
        const image = exampleBatch.xs.slice([i, 0], [1, exampleBatch.xs.shape[1]]);
    
        const div = document.createElement('div');
        div.className = 'pred-container';
    
        const canvas = document.createElement('canvas');
        canvas.className = 'prediction-canvas';
        draw(image.flatten(), canvas);
    
        const pred = document.createElement('div');
    
        const prediction = predictions[i];
        const label = labels[i];
        const correct = prediction === label;

        pred.className = `pred ${(correct ? 'pred-correct' : 'pred-incorrect')}`;
        pred.innerText = `pred: ${prediction} [${label}]`;
    
        div.appendChild(pred);
        div.appendChild(canvas);
    
        imagesElement.appendChild(div);
        if(correct) productivity++;
    }

    productivityString.innerHTML = `Продуктивность модели на данной выборке: <b>${productivity}%</b>`;
}
