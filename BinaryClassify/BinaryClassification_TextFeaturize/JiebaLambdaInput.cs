using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Text;

namespace BinaryClassification_TextFeaturize
{
    public class JiebaLambdaInput
    {
        public string Text { get; set; }
    }

    public class JiebaLambdaOutput
    {
        public string JiebaText { get; set; }
    }

    [CustomMappingFactoryAttribute("LoadJiebaTextAction")]
    public class JiebaLambda : CustomMappingFactory<JiebaLambdaInput, JiebaLambdaOutput>
    {
        static int Count = 0;
        public static void MyAction(JiebaLambdaInput input, JiebaLambdaOutput output)
        {
            JiebaNet.Segmenter.JiebaSegmenter jiebaSegmenter = new JiebaNet.Segmenter.JiebaSegmenter();
            output.JiebaText = string.Join(" ", jiebaSegmenter.Cut(input.Text));

            Count++;
        }

        public override Action<JiebaLambdaInput, JiebaLambdaOutput> GetMapping()
              => MyAction;
    }
}
