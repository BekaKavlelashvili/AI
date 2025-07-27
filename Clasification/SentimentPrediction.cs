using Microsoft.ML.Data;

namespace Classification;

public partial class Program
{
    public class SentimentPrediction
    {
        [ColumnName("Score")]
        public float SentimentScore { get; set; }

        public bool IsPositiveSentiment => SentimentScore < 0.5f;
    }
}
