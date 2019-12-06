using Jc.Core.Helper;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using static Microsoft.ML.DataOperationsCatalog;

namespace MulticlassClassification_Mnist_Useful
{
    class Program
    {
        static readonly string assetsFolder = @"E:\Ps WorkFile\MachineLearning\MySample\ML_Assets\MNIST";
        static readonly string trainTagsPath = Path.Combine(assetsFolder, "train_tags.tsv");
        static readonly string trainDataFolder = Path.Combine(assetsFolder, "train");
        static readonly string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data", "SDCA-Model.zip");

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
            MLContext mlContext = new MLContext();

            // STEP 1: 准备数据
            IDataView fulldata = mlContext.Data.LoadFromTextFile<InputData>(path: trainTagsPath, separatorChar: '\t', hasHeader: false);
            TrainTestData trainTestData = mlContext.Data.TrainTestSplit(fulldata, testFraction: 0.1);
            IDataView trainData = trainTestData.TrainSet;
            IDataView testData = trainTestData.TestSet;


            #region 训练 变换模型
            //训练
            //Estimator 评估    Transforms 变形,变换
            // STEP 2: 配置数据处理管道        
            IEstimator<ITransformer> dataProcessPipeline = 
                mlContext.Transforms.CustomMapping(new LoadImageConversion().GetMapping(), contractName: "LoadImageConversionAction")
               .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", "Number", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue))
               .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: "FeaturesNormalizedByMeanVar", inputColumnName: "ImagePixels"));
            //加载DebugConversion数据读取 用于显示进度
            //将Number转为输出Label
            //将 PixelValues + DebugFeature 组合为Features

            // STEP 3: 配置训练算法 (using a maximum entropy classification model trained with the L-BFGS method)
            IEstimator<ITransformer> trainer = mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "Label", featureColumnName: "FeaturesNormalizedByMeanVar");
            IEstimator<ITransformer> trainingPipeline = dataProcessPipeline.Append(trainer)
                 .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictNumber", "Label"));
            //将输出Label转换为Number

            // STEP 4: 训练模型使其与数据集拟合
            Console.WriteLine("=============== Train the model fitting to the DataSet ===============");

            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();

            LoadImageConversion.InitConversion(trainDataFolder, 0.9);
            ITransformer model = trainingPipeline.Fit(trainData);

            stopWatch.Stop();
            Console.WriteLine($"Used time : { DateHelper.DateDif(stopWatch.Elapsed)}");
            #endregion

            #region 评估

            // STEP 5:评估模型的准确性
            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            LoadImageConversion.InitConversion(trainDataFolder, 0.1);
            IDataView predictions = model.Transform(testData);
            MulticlassClassificationMetrics metrics = mlContext.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
            PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);
            DebugData(mlContext, predictions);

            // STEP 6:保存模型              
            //mlContext.ComponentCatalog.RegisterAssembly(typeof(LoadImageConversion).Assembly);
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
            mlContext.ComponentCatalog.RegisterAssembly(typeof(LoadImageConversion).Assembly);
            ITransformer model = mlContext.Model.Load(modelPath, out var inputSchema);

            //创建预测引擎
            PredictionEngine<InputData, OutPutData> predictionEngine = mlContext.Model.CreatePredictionEngine<InputData, OutPutData>(model);

            Console.ForegroundColor = ConsoleColor.Red;

            Console.WriteLine("===== Test =====");
            DirectoryInfo TestFolder = new DirectoryInfo(Path.Combine(assetsFolder, "test"));
            int count = 0;
            int success = 0;
            foreach (var image in TestFolder.GetFiles())
            {
                count++;

                InputData img = new InputData()
                {
                    FileName = image.Name
                };
                OutPutData result = predictionEngine.Predict(img);

                if (int.Parse(image.Name.Substring(0, 1)) == result.GetPredictResult())
                {
                    success++;
                }

                if (count % 100 == 1)
                {
                    Console.WriteLine($"Current Source={img.FileName},PredictResult={result.GetPredictResult()},Success rate={success * 100 / count}%");
                }
            }

            Console.ResetColor();
        }

        private static void DebugData(MLContext mlContext, IDataView predictions)
        {
            List<string> loadedModelOutputColumnNames = predictions.Schema.Where(col => !col.IsHidden).Select(col => col.Name).ToList();
            foreach (string column in loadedModelOutputColumnNames)
            {
                Console.WriteLine($"loadedModelOutputColumnNames:{ column }");
            }
        }

        public static void PrintMultiClassClassificationMetrics(string name, MulticlassClassificationMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Metrics for {name} multi-class classification model   ");
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

    class InputData
    {
        [LoadColumn(0)]
        public string FileName;

        [LoadColumn(1)]
        public string Number;

        [LoadColumn(1)]
        public float Serial;
    }

    class OutPutData : InputData
    {
        public string PredictNumber;
        public string ImagePath;
        public float[] ImagePixels;
        public float[] Score;
        public int GetPredictResult()
        {
            float max = 0;
            int index = 0;
            for (int i = 0; i < Score.Length; i++)
            {
                if (Score[i] > max)
                {
                    max = Score[i];
                    index = i;
                }
            }
            return index;
        }

        public void PrintToConsole()
        {
            Console.WriteLine($"ImagePath={ImagePath},Number={Number},PredictNumber={PredictNumber}");

            int PredictResult = GetPredictResult();
            Console.WriteLine($"PredictResult={PredictResult}");

            Console.Write($"ImagePixels.Length={ImagePixels.Length},ImagePixels=[");
            for (int i = 0; i < ImagePixels.Length; i++)
            {
                Console.Write($"{ImagePixels[i]},");
            }
            Console.WriteLine("]");

            Console.Write($"Score.Length={Score.Length},Score=[");
            for (int i = 0; i < Score.Length; i++)
            {
                Console.Write($"{Score[i]},");
            }
            Console.WriteLine("]");
        }
    }
}
