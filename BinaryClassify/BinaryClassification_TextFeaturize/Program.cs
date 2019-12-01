using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using static Microsoft.ML.DataOperationsCatalog;

namespace BinaryClassification_TextFeaturize
{
    class Program
    {
        static readonly string DataPath = Path.Combine(Environment.CurrentDirectory, "Data", "meeting_data_full.csv");
        static readonly string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data", "fastmodel.zip");

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
            //准备数据

            IDataView fulldata = mlContext.Data.LoadFromTextFile<MeetingInfo>(DataPath, separatorChar: ',', hasHeader: false);

            TrainTestData trainTestData = mlContext.Data.TrainTestSplit(fulldata, testFraction: 0.15);

            IDataView trainData = trainTestData.TrainSet;
            IDataView testData = trainTestData.TestSet;

            #region 训练 变换模型
            //训练 
            //Estimator 评估    Transforms 变形,变换
            //数据处理管道
            IEstimator<ITransformer> dataProcessPipeline =
                mlContext.Transforms.CustomMapping(new JiebaLambda().GetMapping(), contractName: "LoadJiebaTextAction")
                //mlContext.Transforms.CustomMapping<JiebaLambdaInput, JiebaLambdaOutput>(mapAction: JiebaLambda.MyAction, contractName: "LoadJiebaTextAction")
                .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "JiebaText"));

            //快速决策树
            IEstimator<ITransformer> trainer = mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features");
            
            //扩展决策树 转换器
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
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label");
            
            Console.WriteLine($"Evalution Accuracy: {metrics.Accuracy:P2}");

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

            mlContext.ComponentCatalog.RegisterAssembly(typeof(JiebaLambda).Assembly);
            ITransformer model = mlContext.Model.Load(modelPath, out var inputSchema);

            //创建预测引擎
            PredictionEngine<MeetingInfo, PredictionResult> predictionEngine = mlContext.Model.CreatePredictionEngine<MeetingInfo, PredictionResult>(model);
            
            Console.ForegroundColor = ConsoleColor.Red;

            //预测1
            MeetingInfo sampleStatement1 = new MeetingInfo { Text = "支委会。" };
            PredictionResult prediction1 = predictionEngine.Predict(sampleStatement1);

            //if (prediction1.Score < 0)
            //{
            //    prediction1.Score = prediction1.Score * (-1);
            //    prediction1.Probability = 1 - prediction1.Probability;
            //}
            //prediction1.Probability *= 100;
            Console.WriteLine($"{sampleStatement1.Text}");
            Console.WriteLine($"Predict Result :{prediction1.PredictedLabel}     Probability:{prediction1.Probability}    Score:{prediction1.Score}");

            //预测2
            MeetingInfo sampleStatement2 = new MeetingInfo { Text = "开展新时代中国特色社会主义思想三十讲党员答题活动。" };
            PredictionResult prediction2 = predictionEngine.Predict(sampleStatement2);
            //if (prediction2.Score < 0)
            //{
            //    prediction2.Score = prediction2.Score * (-1);
            //    prediction2.Probability = 1 - prediction2.Probability;
            //}
            //prediction2.Probability *= 100;
            Console.WriteLine($"{prediction2.Text}");
            Console.WriteLine($"Predict Result :{prediction2.PredictedLabel}     Probability:{prediction2.Probability}    Score:{prediction2.Score}");
            
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


    public class MeetingInfo
    {
        [LoadColumn(0)]
        public bool Label { get; set; }
        [LoadColumn(1)]
        public string Text { get; set; }
    }

    public class PredictionResult : MeetingInfo
    {
        public string JiebaText { get; set; }
        public float[] Features { get; set; }
        public bool PredictedLabel;
        public float Score;
        public float Probability;
    }
}
