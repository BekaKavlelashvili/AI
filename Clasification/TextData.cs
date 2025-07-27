using Microsoft.ML.Data;

namespace Classification;

public partial class Program
{
    public class TextData
    {
        [LoadColumn(0)]
        public string Text { get; set; }
    }
}
