using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Text;

namespace MulticlassClassification_Mnist_Useful
{
    public class DebugConversionInput
    {
        public float Serial { get; set; }
    }

    public class DebugConversionOutput
    {
        public float DebugFeature { get; set; }
    }

    [CustomMappingFactoryAttribute("DebugConversionAction")]
    public class DebugConversion : CustomMappingFactory<DebugConversionInput, DebugConversionOutput>
    {
        static long Count = 0;
        static long TotalCount = 0;

        public void CustomAction(DebugConversionInput input, DebugConversionOutput output)
        {
            output.DebugFeature = 1.0f;
            Count++;
            if (Count / 10000 > TotalCount)
            {
                TotalCount = Count / 10000;
                Console.WriteLine($"DebugConversion.CustomAction's debug info.TotalCount={TotalCount}0000 ");
            }
        }

        public override Action<DebugConversionInput, DebugConversionOutput> GetMapping()
              => CustomAction;
    }
}
