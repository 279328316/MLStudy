using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Text;
using System.Threading;

namespace MulticlassClassification_Mnist_Useful
{
    public class LoadImageConversionInput
    {
        public string FileName { get; set; }
    }

    public class LoadImageConversionOutput
    {
        [VectorType(400)]
        public float[] ImagePixels { get; set; }

        public string ImagePath;
    }

    [CustomMappingFactoryAttribute("LoadImageConversionAction")]
    public class LoadImageConversion : CustomMappingFactory<LoadImageConversionInput, LoadImageConversionOutput>
    {
        static long count = 0;
        static long totalCount = 0;
        static long percent = 0;
        static string trainDataFolder = @"D:\StepByStep\Blogs\ML_Assets\MNIST\train";

        public static void InitConversion(string dataFolder,double fraction = 1.0d, string searchPattern = "", SearchOption searchOption = SearchOption.TopDirectoryOnly)
        {
            if (Directory.Exists(dataFolder))
            {
                trainDataFolder = dataFolder;
                long fileAmount = Directory.GetFiles(trainDataFolder, searchPattern, searchOption).Length;
                totalCount = (long)(fileAmount * fraction);
                count = 0;
                percent = 0;
            }
            else
            {
                throw new Exception("初始化的文件夹不存在");
            }
        }

        private void CustomAction(LoadImageConversionInput input, LoadImageConversionOutput output)
        {
            string ImagePath = Path.Combine(trainDataFolder, input.FileName);
            output.ImagePath = ImagePath;

            Bitmap bmp = Image.FromFile(ImagePath) as Bitmap;

            output.ImagePixels = new float[400];
            for (int x = 0; x < 20; x++)
                for (int y = 0; y < 20; y++)
                {
                    var pixel = bmp.GetPixel(x, y);
                    var gray = (pixel.R + pixel.G + pixel.B) / 3 / 16;
                    output.ImagePixels[x + y * 20] = gray;
                }
            bmp.Dispose();

            count++;
            if (count % 1000 == 0)
            {
                Console.Write($"LoadedCount={count}");
                if (totalCount > 0)
                {
                    percent = count * 100 / totalCount;
                    Console.WriteLine($"    Progress : {percent}%");
                }
            }
            Thread.Sleep(1);
        }

        public override Action<LoadImageConversionInput, LoadImageConversionOutput> GetMapping()
              => CustomAction;
    }
}
