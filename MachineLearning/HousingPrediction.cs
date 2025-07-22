using Microsoft.ML.Data;

namespace MachineLearning;

public class HousingPrediction
{
    [ColumnName("Score")]
    public float Price { get; set; }
}