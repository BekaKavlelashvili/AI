using Microsoft.ML.Data;

namespace Classification;

public class MoviewReview
{
    [LoadColumn(0)]
    public string Text { get; set; }

    [LoadColumn(1)]
    [ColumnName("Label")]
    public bool Sentiment { get; set; }
}