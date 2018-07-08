package cl.usach.rn.mlp;

import java.io.File;
import java.io.FileReader;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

public class AnalisisZoologicoUCI {

	public static void main(String[] args) {
		ClassLoader cargadorContexto = Thread.currentThread().getContextClassLoader();

		try {
			// Lectura del set de entrenamiento y prueba. Se cierran los archivos inmediatamente, una vez leídas las instancias.
			FileReader lectorEntrenamiento = new FileReader(new File(cargadorContexto.getResource("zoo.arff").toURI()));
			FileReader lectorPrueba = new FileReader(new File(cargadorContexto.getResource("zoo.arff").toURI()));
			Instances entrenamiento = new Instances(lectorEntrenamiento);
			Instances prueba = new Instances(lectorPrueba);
			lectorEntrenamiento.close();
			lectorPrueba.close();
			
			// Establecemos la columna que tiene la clase
			entrenamiento.setClassIndex(entrenamiento.numAttributes() - 1);
			prueba.setClassIndex(prueba.numAttributes() - 1);

			// Perceptrón multicapa.
			MultilayerPerceptron mlp = new MultilayerPerceptron();

			// Parámetros:
			mlp.setLearningRate(0.1);
			mlp.setMomentum(0.1);
			mlp.setTrainingTime(20000);
			mlp.setSeed(0);
			mlp.setValidationSetSize(0);
			mlp.setValidationThreshold(20);
			mlp.setHiddenLayers("4");

			// Se construye el clasificador
			mlp.buildClassifier(entrenamiento);

			// Se evalúa el modelo con el grupo de prueba
			Evaluation eval = new Evaluation(entrenamiento);
			eval.evaluateModel(mlp, prueba);
			
			// Imprimir todos los resultados.
			System.out.println(eval.toSummaryString());
			System.out.println(eval.toMatrixString());
			System.out.println(mlp.toString());
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}
}
