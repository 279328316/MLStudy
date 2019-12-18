using Jc.Core.Helper;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Diagnostics;
using System.IO;
using static Microsoft.ML.DataOperationsCatalog;

namespace TensorFlow_ImageClassification
{
    class Program
    {
        static readonly string assetsFolder = @"E:\Ps WorkFile\MachineLearning\MySample\ML_Assets";
        static readonly string trainDataFolder = Path.Combine(assetsFolder, "ImageClassification", "train");
        static readonly string trainTagsPath = Path.Combine(assetsFolder, "ImageClassification", "train_tags.tsv");
        static readonly string testDataFolder = Path.Combine(assetsFolder, "ImageClassification", "test");
        static readonly string inceptionPb = Path.Combine(assetsFolder, "TensorFlow", "tensorflow_inception_graph.pb");
        static readonly string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "MLModel", "imageClassifier.zip");

        //配置用常量
        private struct ImageNetSettings
        {
            public const int imageHeight = 224;
            public const int imageWidth = 224;
            public const float mean = 117;
            public const float scale = 1;
            public const bool channelsLast = true;
        }

        static void Main(string[] args)
        {
            TrainAndSave();

            LoadAndPrediction();

            Console.WriteLine("Press any to exit!");
            Console.ReadKey();
        }



        /// <summary>
        /// 训练,保存模型
        /// </summary>
        static void TrainAndSave()
        {
            MLContext mlContext = new MLContext(seed: 1);

            // STEP 1: 准备数据
            IDataView fulldata = mlContext.Data.LoadFromTextFile<ImageNetData>(path: trainTagsPath, separatorChar: '\t', hasHeader: false);
            TrainTestData trainTestData = mlContext.Data.TrainTestSplit(fulldata, testFraction: 0.1);
            IDataView trainData = trainTestData.TrainSet;
            IDataView testData = trainTestData.TestSet;


            #region 训练 变换模型
            //训练
            //Estimator 评估    Transforms 变形,变换
            // STEP 2: 配置数据处理管道
            IEstimator<ITransformer> dataProcessPipeline =
                mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelTokey", inputColumnName: "Label")
                .Append(mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: trainDataFolder, inputColumnName: nameof(ImageNetData.ImagePath)))
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: ImageNetSettings.imageWidth, imageHeight: ImageNetSettings.imageHeight, inputColumnName: "input"))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: ImageNetSettings.channelsLast, offsetImage: ImageNetSettings.mean))
                .Append(mlContext.Model.LoadTensorFlowModel(inceptionPb).
                     ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelTokey", featureColumnName: "softmax2_pre_activation"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
                .AppendCacheCheckpoint(mlContext);
            //加载DebugConversion数据读取 用于显示进度
            //将Number转为输出Label                                  

            // STEP 4: 训练模型使其与数据集拟合
            Console.WriteLine("=============== Train the model fitting to the DataSet ===============");

            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();

            LogHelper.WriteAppLog("==================训练====================");
            ITransformer model = dataProcessPipeline.Fit(trainData);

            stopWatch.Stop();
            Console.WriteLine($"Used time : { DateHelper.DateDif(stopWatch.Elapsed)}");
            #endregion

            #region 评估
            Console.WriteLine("===== Evaluate model =======");
            var evaData = model.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(evaData, labelColumnName: "LabelTokey", predictedLabelColumnName: "PredictedLabel");
            PrintMultiClassClassificationMetrics(metrics);

            //STEP 5：保存模型
            Console.WriteLine("====== Save model to local file =========");
            mlContext.Model.Save(model, trainData.Schema, modelPath);
            Console.WriteLine("The model is saved to {0}", modelPath);

            #endregion
        }

        /// <summary>
        /// 加载变换模型并预测
        /// </summary>
        static void LoadAndPrediction()
        {
            MLContext mlContext = new MLContext();

            //加载变换模型
            ITransformer model = mlContext.Model.Load(modelPath, out var inputSchema);

            //创建预测引擎
            PredictionEngine<ImageNetData, ImageNetPrediction> predictionEngine = mlContext.Model.CreatePredictionEngine<ImageNetData, ImageNetPrediction>(model);

            Console.ForegroundColor = ConsoleColor.Red;

            Console.WriteLine("===== Test =====");

            DirectoryInfo testdir = new DirectoryInfo(testDataFolder);
            foreach (FileInfo jpgfile in testdir.GetFiles("*.jpg"))
            {
                ImageNetData image = new ImageNetData();
                image.ImagePath = jpgfile.FullName;
                var pred = predictionEngine.Predict(image);

                Console.WriteLine($"Filename:{jpgfile.Name}:\tPredict Result:{pred.PredictedLabelValue}");
            }

            Console.ResetColor();
        }

        public static void PrintMultiClassClassificationMetrics(MulticlassClassificationMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Metrics for L-BFGS  multi-class classification model   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"    AccuracyMacro = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    AccuracyMicro = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 1 = {metrics.PerClassLogLoss[0]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 2 = {metrics.PerClassLogLoss[1]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 3 = {metrics.PerClassLogLoss[2]:0.####}, the closer to 0, the better");
            Console.WriteLine($"************************************************************");
        }

    }

    public class ImageNetData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;
    }

    public class ImageNetPrediction
    {
        //public float[] Score;
        public string PredictedLabelValue;
    }
}
