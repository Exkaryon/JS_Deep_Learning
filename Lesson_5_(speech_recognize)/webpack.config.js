const path = require('path');
const HTMLWebpackPlugin = require('html-webpack-plugin'); 
const CopyWebpackPlugin = require('copy-webpack-plugin'); 

const isDev = process.env.NODE_ENV === 'development';
const isProd = !isDev;
console.log('IS DEV:', isDev)
const filename = ext => isDev ? `[name].${ext}` : `[name].[hash].${ext}`;

module.exports = {
    context: path.resolve(__dirname, 'src'),            // Папка контекст ./src/
    mode: 'development',                                // Режим сборки
    entry: {                                            // Точки входа
        main: './index.js',
    },
    output: {
        filename: filename('js'),
        path: path.resolve(__dirname, 'dist'),
        clean: true                                     // Нововведение WP: Заменяет плагин CleanWebpackPlugin
    },
    devServer: {
        port: 1234,
        watchFiles: ['src/**/*.html', 'public/**/*', 'src/**/*.js', 'src/*.html', 'src/*.js', 'src/**/*.css', 'src/*.css'],     // маски отслеживаемых файлов для обновления страницы
    },
    devtool: isDev ? 'source-map' : '',
    plugins: [
        new HTMLWebpackPlugin({                         // Плагин для обработки HTML-шаблона
            template: './index.html',
            minify: {
                collapseWhitespace: isProd,
            }
        }),
        new CopyWebpackPlugin({
            patterns: [
                {
                    from: path.resolve(__dirname, 'src/favicon.ico'),
                    to: path.resolve(__dirname, 'dist'),
                },
                {
                    from: path.resolve(__dirname, 'dataset/msc_dataset.json'),
                    to: path.resolve(__dirname, 'dist/dataset'),
                },
                /* Раскоментить, если требуется скопировать датасет в dist при сборке.*/
                {
                    from: path.resolve(__dirname, 'dataset/mini_speech_commands'),
                    to: path.resolve(__dirname, 'dist/dataset'),
                },
                {
                    from: path.resolve(__dirname, 'dataset/msc_ready_data.json'),
                    to: path.resolve(__dirname, 'dist/dataset'),
                },
              
            ],
        }),
    ],
    module: {
        rules: [
            {
                test: /\.css$/,
                use: ['style-loader', 'css-loader']
            },
            {
                test: /\.csv$/,
                use:['csv-loader']
            },
        ]
    }
}