using Microsoft.ML;

namespace Classification;

public partial class Program
{
    public static void Main(string[] args)
    {
        string modelPath = "sentiment_model.zip";
        string testDataPath = "movieReviewsTesting.csv";

        MLContext mlContext = new MLContext();

        ITransformer model;
        using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            model = mlContext.Model.Load(stream, out var modelInputSchema);
        }

        IDataView testData = mlContext.Data.LoadFromTextFile<TextData>(testDataPath, separatorChar: ',', hasHeader: true);
        var predictor = mlContext.Model.CreatePredictionEngine<TextData, SentimentPrediction>(model);

        var testDataEnumerable = mlContext.Data.CreateEnumerable<TextData>(testData, reuseRowObject: false).ToList();

        foreach (var data in testDataEnumerable)
        {
            var prediction = predictor.Predict(data);
            Console.WriteLine($"Text: {data.Text} | Positive Sentiment: {prediction.IsPositiveSentiment}");
        }
    }
}
