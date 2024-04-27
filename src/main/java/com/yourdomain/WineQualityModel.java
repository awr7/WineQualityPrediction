package com.yourdomain;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.CrossValidator;
import java.util.Arrays;

public class WineQualityModel {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
            .appName("Wine Quality Prediction")
            .getOrCreate();

        Dataset<Row> trainingData = spark.read().format("csv")
            .option("inferSchema", "true")
            .option("header", "true")
            .option("delimiter", ";")
            .load(args[0]);

        System.out.println("Original Columns: " + Arrays.toString(trainingData.columns()));
        for (String colName : trainingData.columns()) {
            trainingData = trainingData.withColumnRenamed(colName, colName.replace("\"", "").trim());
        }
        System.out.println("Sanitized Columns: " + Arrays.toString(trainingData.columns()));

        String[] inputCols = {"fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol"};
        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(inputCols)
            .setOutputCol("features");

        Dataset<Row> trainingDataTransformed = assembler.transform(trainingData);

        LogisticRegression lr = new LogisticRegression()
            .setLabelCol("quality")
            .setFeaturesCol("features");

        ParamGridBuilder paramGrid = new ParamGridBuilder()
            .addGrid(lr.regParam(), new double[]{0.01, 0.1, 1.0})
            .addGrid(lr.maxIter(), new int[]{100, 200});

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("quality")
            .setPredictionCol("prediction")
            .setMetricName("f1");

        CrossValidator cv = new CrossValidator()
            .setEstimator(lr)
            .setEvaluator(evaluator)
            .setEstimatorParamMaps(paramGrid.build())
            .setNumFolds(3);

        CrossValidatorModel model = cv.fit(trainingDataTransformed);

        Dataset<Row> validationData = spark.read().format("csv")
            .option("inferSchema", "true")
            .option("header", "true")
            .option("delimiter", ";")
            .load(args[1]);
        for (String colName : validationData.columns()) {
            validationData = validationData.withColumnRenamed(colName, colName.replace("\"", "").trim());
        }

        Dataset<Row> validationDataTransformed = assembler.transform(validationData);
        Dataset<Row> predictions = model.transform(validationDataTransformed);

        double f1Score = evaluator.evaluate(predictions);
        System.out.println("F1 Score = " + f1Score);

        spark.stop();
    }
}
