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

namespace MulticlassClassification_Mnist
{
    class Program
    {
        static readonly string dataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data", "optdigits-full.csv");
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
            IDataView fulldata = mlContext.Data.LoadFromTextFile(path: dataPath,
                    columns: new[]
                    {
                        new TextLoader.Column("Serial", DataKind.Single, 0),
                        new TextLoader.Column("PixelValues", DataKind.Single, 1, 64),
                        new TextLoader.Column("Number", DataKind.Single, 65)
                    },
                    hasHeader: true,
                    separatorChar: ','
                    );

            TrainTestData trainTestData = mlContext.Data.TrainTestSplit(fulldata, testFraction: 0.2);

            IDataView trainData = trainTestData.TrainSet;
            IDataView testData = trainTestData.TestSet;

            #region 训练 变换模型
            //训练
            //Estimator 评估    Transforms 变形,变换
            //STEP 2: 配置数据处理管道 
            IEstimator<ITransformer> dataProcessPipeline = mlContext.Transforms.CustomMapping(new DebugConversion().GetMapping(), contractName: "DebugConversionAction")
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", "Number", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue))
                .Append(mlContext.Transforms.Concatenate("Features", new string[] { "PixelValues", "DebugFeature" }));
            //加载DebugConversion数据读取 用于显示进度
            //将Number转为输出Label
            //将 PixelValues + DebugFeature 组合为Features

            //STEP 3: 配置训练算法
            IEstimator<ITransformer> trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");
            IEstimator<ITransformer> trainingPipeline = dataProcessPipeline.Append(trainer)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("Number", "Label"));
            //将输出Label转换为Number

            // STEP 4: 训练模型使其与数据集拟合
            Console.WriteLine("=============== Train the model fitting to the DataSet ===============");

            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();

            ITransformer model = trainingPipeline.Fit(trainData);

            stopWatch.Stop();
            Console.WriteLine($"Used time : { DateHelper.DateDif(stopWatch.Elapsed)}");
            #endregion

            #region 评估

            // STEP 5:评估模型的准确性
            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            IDataView predictions = model.Transform(testData);
            MulticlassClassificationMetrics metrics = mlContext.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "Number", scoreColumnName: "Score");
            PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);
            DebugData(mlContext, predictions);

            // STEP 6:保存模型              
            //mlContext.ComponentCatalog.RegisterAssembly(typeof(DebugConversion).Assembly);
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
            mlContext.ComponentCatalog.RegisterAssembly(typeof(DebugConversion).Assembly);
            ITransformer model = mlContext.Model.Load(modelPath, out var inputSchema);

            //创建预测引擎
            PredictionEngine<InputData, OutPutData> predictionEngine = mlContext.Model.CreatePredictionEngine<InputData, OutPutData>(model);

            Console.ForegroundColor = ConsoleColor.Red;

            //num 1
            InputData MNIST1 = new InputData()
            {
                PixelValues = new float[] { 0, 0, 0, 0, 14, 13, 1, 0, 0, 0, 0, 5, 16, 16, 2, 0, 0, 0, 0, 14, 16, 12, 0, 0, 0, 1, 10, 16, 16, 12, 0, 0, 0, 3, 12, 14, 16, 9, 0, 0, 0, 0, 0, 5, 16, 15, 0, 0, 0, 0, 0, 4, 16, 14, 0, 0, 0, 0, 0, 1, 13, 16, 1, 0 }
            };
            OutPutData resultprediction1 = predictionEngine.Predict(MNIST1);
            resultprediction1.PrintToConsole();

            //num 7
            InputData MNIST2 = new InputData()
            {
                PixelValues = new float[] { 0, 0, 1, 8, 15, 10, 0, 0, 0, 3, 13, 15, 14, 14, 0, 0, 0, 5, 10, 0, 10, 12, 0, 0, 0, 0, 3, 5, 15, 10, 2, 0, 0, 0, 16, 16, 16, 16, 12, 0, 0, 1, 8, 12, 14, 8, 3, 0, 0, 0, 0, 10, 13, 0, 0, 0, 0, 0, 0, 11, 9, 0, 0, 0 }
            };
            OutPutData resultprediction2 = predictionEngine.Predict(MNIST2);
            resultprediction2.PrintToConsole();
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
        public float Serial;
        [VectorType(64)]
        public float[] PixelValues;
        public float Number;
    }

    class OutPutData : InputData
    {
        public float[] Score;

        public void PrintToConsole()
        {
            Console.WriteLine($"Predicted probability:     zero:  {Score[0]:0.####}");
            Console.WriteLine($"                           One :  {Score[1]:0.####}");
            Console.WriteLine($"                           two:   {Score[2]:0.####}");
            Console.WriteLine($"                           three: {Score[3]:0.####}");
            Console.WriteLine($"                           four:  {Score[4]:0.####}");
            Console.WriteLine($"                           five:  {Score[5]:0.####}");
            Console.WriteLine($"                           six:   {Score[6]:0.####}");
            Console.WriteLine($"                           seven: {Score[7]:0.####}");
            Console.WriteLine($"                           eight: {Score[8]:0.####}");
            Console.WriteLine($"                           nine:  {Score[9]:0.####}");
        }
    }
}
