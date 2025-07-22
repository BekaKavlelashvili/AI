using MachineLearning;
using Microsoft.ML;

public class Program
{
    static void Main(string[] args)
    {
        MLContext mlContext = new MLContext();
        IDataView data = mlContext.Data.LoadFromTextFile<HousingData>("housing-data.csv", separatorChar: ',', hasHeader: true);

        // Just introduction to ML.NET
        introToML(data, mlContext);

        Console.WriteLine();

        // Transform the data
        transformData(data, mlContext);
    }

    static void introToML(IDataView data, MLContext mlContext)
    {
        Console.WriteLine("==================== Introduction to ML.NET ====================");

        string[] featureColumns = { "SquareFeet", "Bedrooms" };
        string labelColumn = "Price";

        var pipeline = mlContext.Transforms.Concatenate("Features", featureColumns)
            .Append(mlContext.Regression.Trainers.FastTree(labelColumnName: labelColumn));

        var model = pipeline.Fit(data);

        var prediction = model.Transform(data);
        var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: labelColumn);
        Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError}");
        Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");

    }

    static void transformData(IDataView data, MLContext mlContext)
    {
        Console.WriteLine("==================== Transform Data ====================");

        var dataPipeline = mlContext.Transforms.Conversion.ConvertType("SquareFeet", outputKind: Microsoft.ML.Data.DataKind.Single)
            .Append(mlContext.Transforms.NormalizeMinMax("SquareFeet"))
            .Append(mlContext.Transforms.Concatenate("Features", "SquareFeet", "Bedrooms"))
            .Append(mlContext.Transforms.Categorical.OneHotEncoding("Neighborhood"));

        var transformedData = dataPipeline.Fit(data).Transform(data);
        var transformedDataEnumerable = mlContext.Data.CreateEnumerable<TransformedHousingData>(transformedData, reuseRowObject: false).ToList();
        foreach (var item in transformedDataEnumerable)
        {
            Console.WriteLine($"SquareFeet: {item.SquareFeet}, Bedroom: {item.Bedrooms}, Price: {item.Price}, Features: [{string.Join(", ", item.Features)}], Neighborhood: [{string.Join(", ", item.Neighborhood)}]");
        }
    }
}
