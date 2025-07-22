using Microsoft.ML.Data;

public partial class Program
{
    public class DataPoint
    {
        [LoadColumn(0)]
        public float Feature1 { get; set; }

        [LoadColumn(1)]
        public float Feature2 { get; set; }

        [LoadColumn(2)]
        public bool Label { get; set; }
    }
}