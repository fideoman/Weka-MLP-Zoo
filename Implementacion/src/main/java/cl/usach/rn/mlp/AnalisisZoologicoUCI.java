package cl.usach.rn.mlp;

import java.io.File;
import java.io.FileReader;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.Utils;

public class AnalisisZoologicoUCI {

	public static void main(String[] args) {
		ClassLoader cargadorContexto = Thread.currentThread().getContextClassLoader();

		try {
			FileReader trainreader = new FileReader(new File(cargadorContexto.getResource("zoo.arff").toURI()));
			FileReader testreader = new FileReader(new File(cargadorContexto.getResource("zoo.arff").toURI()));

			Instances train = new Instances(trainreader);
			Instances test = new Instances(testreader);
			train.setClassIndex(train.numAttributes() - 1);
			test.setClassIndex(test.numAttributes() - 1);

			MultilayerPerceptron mlp = new MultilayerPerceptron();
			System.out.println("Parámetros: -L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H 4");
			mlp.setOptions(Utils.splitOptions("-L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H 4"));

			mlp.buildClassifier(train);

			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(mlp, test);
			System.out.println(eval.toSummaryString("\nResultados\n==========\n", false));
			trainreader.close();
			testreader.close();

		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}
}
