namespace MachineLearning;

public class TransformedHousingData
{
    public float SquareFeet { get; set; }

    public float Bedrooms { get; set; }

    public float Price { get; set; }

    // This will hold the concatenated features for ML.NET
    public float[] Features { get; set; }

    public float[] Neighborhood { get; set; }

}