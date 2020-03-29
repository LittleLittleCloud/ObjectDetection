﻿// This file was auto-generated by ML.NET Model Builder. 

using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Linq;

namespace ObjectDetection
{
    [CustomMappingFactoryAttribute(nameof(ReshapeTransformer))]
    public class ReshapeTransformer : CustomMappingFactory<Input, Output>
    {
        // This is the custom mapping. We now separate it into a method, so that we can use it both in training and in loading.
        public static void Mapping(Input input, Output output)
        {
            var values = input.Reshape.GetValues().ToArray();

            for (int x = 0; x < values.Count(); x++)
            {
                // normalize from [0, 255] to [0, 1]
                values[x] /= 255;
            };

            output.Reshape = new VBuffer<float>(values.Count(), values);
        }
        // This factory method will be called when loading the model to get the mapping operation.
        public override Action<Input, Output> GetMapping()
        {
            return Mapping;
        }
    }
    public class Input
    {
        [ColumnName("input")]
        [VectorType(3, 800, 600)]
        public VBuffer<float> Reshape;
    }
    public class Output
    {
        [ColumnName("input")]
        [VectorType(1 * 3 * 600 * 800)]
        public VBuffer<float> Reshape;
    }
}