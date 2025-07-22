using Microsoft.ML;
using Microsoft.ML.Data;

public partial class Program
{
    static void Main(string[] args)
    {
        MLContext context = new();
        var data = context.Data.LoadFromTextFile<DataPoint>("data.csv", separatorChar: ',', hasHeader: true);

        var trainTestSplit = context.Data.TrainTestSplit(data, testFraction: 0.2);

        var logisticRegressionPipeline = context.Transforms.Concatenate("Features", "Feature1", "Feature2")
            .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", maximumNumberOfIterations: 100));

        var fastTreePipeline = context.Transforms.Concatenate("Features", "Feature1", "Feature2")
            .Append(context.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", numberOfLeaves: 50, numberOfTrees: 100));

        Console.WriteLine("Training Logistic Regression model...");
        var logisticRegressionModel = logisticRegressionPipeline.Fit(trainTestSplit.TrainSet);

        Console.WriteLine("Training FastTree model...");
        var fastTreeModel = fastTreePipeline.Fit(trainTestSplit.TrainSet);

        Console.WriteLine("Evaluating Logistic Regression model...");
        var logisticRegressionPredictions = logisticRegressionModel.Transform(trainTestSplit.TestSet);
        var logisticRegressionMetrics = context.BinaryClassification.Evaluate(logisticRegressionPredictions);

        evaluateMetrics("Logistic Regression", logisticRegressionMetrics);

        Console.WriteLine("Evaluating FastTree model...");
        var fastTreePredictions = fastTreeModel.Transform(trainTestSplit.TestSet);
        var fastTreeMetrics = context.BinaryClassification.Evaluate(fastTreePredictions);

        evaluateMetrics("FastTree", fastTreeMetrics);

        compareAccuracy(logisticRegressionMetrics, fastTreeMetrics);
    }

    static void evaluateMetrics(string modelName, BinaryClassificationMetrics metrics)
    {
        Console.WriteLine($"{modelName} - Accuracy: {metrics.Accuracy:0.##}");
        Console.WriteLine($"{modelName} - AUC: {metrics.AreaUnderRocCurve:0.##}");
    }

    static void compareAccuracy(CalibratedBinaryClassificationMetrics logisticRegressionMetrics, CalibratedBinaryClassificationMetrics fastTreeMetrics)
    {
        if (logisticRegressionMetrics.Accuracy > fastTreeMetrics.Accuracy)
        {
            Console.WriteLine("Logistic Regression is the best model.");
        }
        else if (fastTreeMetrics.Accuracy > logisticRegressionMetrics.Accuracy)
        {
            Console.WriteLine("FastTree is the best model.");
        }
        else
        {
            Console.WriteLine("Logistic Regression and FastTree are equally as good.");
        }
    }
}