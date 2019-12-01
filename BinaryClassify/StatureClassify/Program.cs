using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using static Microsoft.ML.DataOperationsCatalog;

namespace StatureClassify
{
    enum Result
    {
        Bad = 0,
        Good = 1
    }

    class Program
    {
        static readonly string dataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data", "staure.csv");
        static readonly string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data", "fastmodel.zip");

        static void Main(string[] args)
        {
            GenerateTrainData();

            TrainAndSave();

            LoadAndPrediction();

            Console.WriteLine("Press any to exit!");
            Console.ReadKey();
        }

        /// <summary>
        /// 生成训练数据
        /// </summary>
        static void GenerateTrainData()
        {
            if (File.Exists(dataPath))
            {
                File.Delete(dataPath);
                //return;
            }
            string dirPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data");
            if (!Directory.Exists(dirPath))
            {
                Directory.CreateDirectory(dirPath);
            }

            using (StreamWriter sw = new StreamWriter(dataPath, false))
            {
                Random random = new Random();
                float height, weight;
                Result result;

                sw.WriteLine("Height,Weight,Result");
                for (int i = 0; i < 20000; i++)
                {
                    height = random.Next(150, 195);
                    weight = random.Next(70, 200);

                    if (height >= 170 && weight < 120)
                    {
                        result = Result.Good;
                    }
                    else
                    {
                        result = Result.Bad;
                    }
                    sw.WriteLine($"{height},{weight},{(int)result}");
                }
            }
        }

        /// <summary>
        /// 训练,保存模型
        /// </summary>
        static void TrainAndSave()
        {
            MLContext mlContext = new MLContext();
            //准备数据
            IDataView fulldata = mlContext.Data.LoadFromTextFile<FigureData>(path: dataPath, hasHeader: true, separatorChar: ',');

            TrainTestData trainTestData = mlContext.Data.TrainTestSplit(fulldata, testFraction: 0.2);

            IDataView trainData = trainTestData.TrainSet;
            IDataView testData = trainTestData.TestSet;

            #region 训练 变换模型
            //训练 
            //Estimator 评估    Transforms 变形,变换
            //NormalizeMeanVariance 标准化方差
            IEstimator<ITransformer> dataProcessPipeline = mlContext.Transforms.Concatenate("Features", new[] { "Height", "Weight" })
                .Append(mlContext.Transforms.NormalizeMeanVariance(inputColumnName: "Features", outputColumnName: "FeaturesNormalizedByMeanVar"));
            
            //快速决策树
            IEstimator<ITransformer> trainer = mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Result", featureColumnName: "FeaturesNormalizedByMeanVar");
            
            IEstimator<ITransformer> trainingPipeline = dataProcessPipeline.Append(trainer);

            //训练 变换模型
            ITransformer model = trainingPipeline.Fit(trainData);

            #endregion

            #region 评估
            //评估的过程就是对测试数据集进行批量转换（Transform），
            //转换过的数据集会多出一个“PredictedLabel”的列，这个就是模型评估的结果，
            //逐条将这个结果和实际结果（Result）进行比较，就最终形成了效果评估数据
            
            IDataView predictions = model.Transform(testData);

            //二元分类校准度量
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Result");
            
            PrintBinaryClassificationMetrics(trainer.ToString(), metrics);

            #endregion

            //保存模型
            mlContext.Model.Save(model, trainData.Schema, modelPath);
            Console.WriteLine($"Model file saved to :{modelPath}");
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
            PredictionEngine<FigureData, FigureDatePredicted> predictionEngine = mlContext.Model.CreatePredictionEngine<FigureData, FigureDatePredicted>(model);

            FigureData test = new FigureData();
            test.Weight = 110;
            test.Height = 175;

            FigureDatePredicted prediction = predictionEngine.Predict(test);

            Console.ForegroundColor = ConsoleColor.Red;            
            if(prediction.Score<0)
            {
                prediction.Score = prediction.Score * (-1);
                prediction.Probability = 1 - prediction.Probability;
            }
            prediction.Probability *= 100;
            Console.WriteLine($"Predict Result :{prediction.PredictedLabel}     Probability:{prediction.Probability:#.##}    Score:{prediction.Score:#.##}");
            Console.ResetColor();
        }

        /// <summary>
        /// 打印二元分类校准
        /// </summary>
        /// <param name="name"></param>
        /// <param name="metrics"></param>
        public static void PrintBinaryClassificationMetrics(string name, CalibratedBinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*       Metrics for {name} binary classification model      ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"*       Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"*       Area Under Curve:      {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"*       Area under Precision recall Curve:  {metrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"*       F1Score:  {metrics.F1Score:P2}");
            Console.WriteLine($"*       LogLoss:  {metrics.LogLoss:#.##}");
            Console.WriteLine($"*       LogLossReduction:  {metrics.LogLossReduction:#.##}");
            Console.WriteLine($"*       PositivePrecision:  {metrics.PositivePrecision:#.##}");
            Console.WriteLine($"*       PositiveRecall:  {metrics.PositiveRecall:#.##}");
            Console.WriteLine($"*       NegativePrecision:  {metrics.NegativePrecision:#.##}");
            Console.WriteLine($"*       NegativeRecall:  {metrics.NegativeRecall:P2}");
            Console.WriteLine($"************************************************************");
        }

    }


    public class FigureData
    {
        [LoadColumn(0)]
        public float Height { get; set; }

        [LoadColumn(1)]
        public float Weight { get; set; }

        [LoadColumn(2)]
        public bool Result { get; set; }
    }

    public class FigureDatePredicted : FigureData
    {
        public bool PredictedLabel;
        public float Probability;
        public float Score;
    }
}
