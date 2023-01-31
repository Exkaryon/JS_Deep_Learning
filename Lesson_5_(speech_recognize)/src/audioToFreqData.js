export class SpectrogramCreator {

    fftSize = 256;      // Размер БПФ для анализа в частотной области - определяет высоту спектрограммы. Значение устанавливается из интерфейса.
    bufferSize = 256;   // Размер буфера для аудиокадра - определяет ширину спектрограммы. Значение устанавливается из интерфейса.

    async getFreqDataFromFile(file){
        const audioBuffer = await this.getAudioBufferFromFile(file)
        const offlineAudio = this.createOfflineAudio(audioBuffer);
        const freqData = await this.processor(offlineAudio);
        return freqData;
    }


    ////////////////////////////////////////////
    // Загрузка и буферизация звукового файла //
    ////////////////////////////////////////////
    async getAudioBufferFromFile(file){
        const audioContext = new AudioContext();
        let audioBuffer = null;
        try{
            let response = await fetch(file);
            if(!response.ok) throw new Error(); 
            const responseBuffer = await response.arrayBuffer();                          // responseBuffer содержит в себе длину буфера в байтах, находящихся в памяти. Его можно представить, например, в виде беззнакового 8-битного массива через new Uint8Array();
            audioBuffer = await audioContext.decodeAudioData(responseBuffer);       // Декодирование данных из буфера в аудиобуфер, над которым можно проделывать аудиооперации (прогирывать, анализировать и т.д.)
        }catch(error){
            console.error('Ошибка загрузки файла звука!' + error);
        }
        return audioBuffer;
    }


    ////////////////////////////////////
    // Создание офлайн аудиоконтекста //
    ////////////////////////////////////
    createOfflineAudio(audioBuffer){
        // Создаем новый офлайн контекст аудио. Для OfflineAudioContext звук не воспроизводится аппаратным обеспечением устройства; вместо этого он генерирует его так быстро, как только может, и выводит результат в renderedBuffer.
        const offlineContext = new OfflineAudioContext(
            audioBuffer.numberOfChannels,
            audioBuffer.length,
            audioBuffer.sampleRate
        );

        // Метод createBufferSource() интерфейса BaseAudioContext используется для создания нового объекта AudioBufferSourceNode, который можно использовать для воспроизведения аудиоданных, содержащихся в AudioBuffer объекте.
        const source = offlineContext.createBufferSource();
        source.buffer = audioBuffer;
        source.channelCount = audioBuffer.numberOfChannels;                 // Количество каналов (в данном скрипте подразумеваются аудиофайлы только с одним каналом)

        // Метод createAnalyser() интерфейса BaseAudioContext создает объект AnalyserNode, который можно использовать для предоставления данных о времени и частоте звука и создания визуализаций данных.
        const analyserNode = offlineContext.createAnalyser();
        analyserNode.fftSize = this.fftSize;                                // Размер БПФ для анализа в частотной области. Если fftSize = 256, то в массиве частот будет сождержаться в два раза меньше - 128 элементов.
        analyserNode.smoothingTimeConstant = 0.1                            // Желаемая начальная константа сглаживания. Значение по умолчанию 0.8.
        return {offlineContext, analyserNode, source};
    }



    ////////////////////////////////////////////////////////////////////////
    // Проигрывание офлайн AudioBufferSourceNode и сбор данных о чатсотах //
    ////////////////////////////////////////////////////////////////////////
    async processor({offlineContext, analyserNode, source}){
        const frequencyData = [];

        // Метод createScriptProcessor() интерфейса BaseAudioContext создает ScriptProcessorNode используемый для прямой обработки звука.
        const processor = offlineContext.createScriptProcessor(this.bufferSize, 1, 1);      // bufferSize - Размер буфера определяющий кадр. 256, 512, 1024, 2048, 4096, 8192, 16384. Если он не передан или если значение равно 0, тогда реализация выберет наилучший размер буфера для данной среде, которая будет постоянной степенью 2 на протяжении всего времени существования узла.
        let byteOffset = 0;                                                     // Смещение в байтах для каждой итерации проигрывания временного отрезка (audioBuffer.duration / (audioBuffer.length / 256) = n сек).
        // Выполняется на каждом блоке звука (выборочных кадров) при проигрывании аудио.
        processor.onaudioprocess = (ev) => {    // ev - объект текущего состояния
            const freqData = new Uint8Array(analyserNode.frequencyBinCount, byteOffset);
            analyserNode.getByteFrequencyData(freqData);
            byteOffset += analyserNode.frequencyBinCount;
            frequencyData.push(freqData);
        }

        source.connect(processor);
        processor.connect(offlineContext.destination);
        source.connect(analyserNode);
        // Start the source, other wise start rendering would not process the source
        await source.start(0);
        await offlineContext.startRendering();
        return frequencyData;
    }
}


export default new SpectrogramCreator();




