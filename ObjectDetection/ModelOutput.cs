using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ObjectDetection
{
    public class ModelOutput
    {
        [ColumnName("boxes")]
        public VBuffer<float> Boxes;

        [ColumnName("labels")]
        public VBuffer<long> Labels;

        [ColumnName("scores")]
        public VBuffer<float> Scores;

        private BoundingBox[] BoundingBoxes
        {
            get
            {
                var boundingBoxes = new List<BoundingBox>();

                boundingBoxes = Enumerable.Range(0, this.Labels.Length)
                          .Select((index) =>
                          {
                              var boxes = this.Boxes.GetValues().ToArray();
                              var scores = this.Scores.GetValues().ToArray();
                              var labels = this.Labels.GetValues().ToArray();

                              return new BoundingBox()
                              {
                                  Left = boxes[index * 4],
                                  Top = boxes[index * 4 + 1],
                                  Right = boxes[index * 4 + 2],
                                  Bottom = boxes[index * 4 + 3],
                                  Score = scores[index],
                                  Label = labels[index].ToString(),
                              };
                          }).ToList();
                return boundingBoxes.ToArray();
            }
        }
        public BoundingBox[] GetBoundingBoxes()
        {
            return this.BoundingBoxes;
        }

        public override string ToString()
        {
            var sb = new StringBuilder();

            foreach(var box in this.BoundingBoxes)
            {
                sb.AppendLine(box.ToString());
            }

            return sb.ToString();
        }
    }

    public class BoundingBox
    {
        public float Top;

        public float Left;

        public float Right;

        public float Bottom;

        public string Label;

        public float Score;

        public override string ToString()
        {
            return $"Top: {this.Top}, Left: {this.Left}, Right: {this.Right}, Bottom: {this.Bottom}, Label: {this.Label}, Score: {this.Score}";
        }
    }
}
