using Microsoft.ML.Data;

public partial class Program
{
    public class Prediction
    {
        [ColumnName("Score")]
        public float Score { get; set; }

        [ColumnName("Probability")]
        public float Probability { get; set; }
    }
}