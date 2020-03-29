using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;

namespace ObjectDetection
{
    class Program
    {
        private static string OnnxInput = "input";
        private static string Assets = @"./Assets";
        private static string OnnxModel = Path.Combine(Assets,"model.onnx");


        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            var model = CreateMLNetModel(mlContext, OnnxModel);

            var example = new ModelInput()
            {
                ImagePath = "000001001.png",
            };

            var predictEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

            var result = predictEngine.Predict(example);
            Console.WriteLine(result);
        }

        static ITransformer CreateMLNetModel(MLContext mlContext, string onnxModel)
        {
            var data = mlContext.Data.LoadFromEnumerable(new List<ModelInput>());


            // Define scoring pipeline
            var pipeline = mlContext.Transforms.LoadImages(outputColumnName: OnnxInput, imageFolder: Assets, inputColumnName: nameof(ModelInput.ImagePath))
                            .Append(mlContext.Transforms.ResizeImages(outputColumnName: OnnxInput, imageWidth: 800, imageHeight: 600, inputColumnName: OnnxInput))
                            .Append(mlContext.Transforms.ExtractPixels(outputColumnName: OnnxInput, inputColumnName: OnnxInput))
                            .Append(mlContext.Transforms.CustomMapping<Input, Output>(
                                          (input, output) => ReshapeTransformer.Mapping(input, output),
                                          contractName: nameof(ReshapeTransformer)))
                            .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: onnxModel, outputColumnNames: new[] { "boxes", "labels", "scores" }, inputColumnNames: new[] { OnnxInput }));

            var model = pipeline.Fit(data);

            return model;
        }
    }
}
