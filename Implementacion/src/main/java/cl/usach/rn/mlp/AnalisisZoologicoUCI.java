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
			mlp.setLearningRate(0.1); 				// Tasa de aprendizaje. De 0 a 1. Por defecto: 0.3
			mlp.setMomentum(0.1); 					// Momentum para el algoritmo de retro-propagación. De 0 a 1. Por defecto: 0.2
			mlp.setTrainingTime(8000); 				// Tiempo de entrenamiento, en épocas. Por defecto: 500
			mlp.setValidationSetSize(0); 			/* Tamaño (en porcentaje) del grupo de validación 
														usado para terminar el entrenamiento. 
														Si no es cero, puede pre-calcular número de épocas. 
														De 0 a 100. Por defecto: 0 */
			mlp.setSeed(0); 						// Semilla. El valor para alimentar al generator de números aleatorios. Por defecto: 0
			mlp.setValidationThreshold(20); 		/* El número máximo de errores consecutivos permitido para las pruebas de validación
														antes que la red termine. Valor debe ser mayor que cero. Por defecto: 20*/
			mlp.setGUI(false); 						// Para abrir, o no, una guía gráfica.
			mlp.setAutoBuild(true); 				// En el caso de elegir GUI, ésta opción indica si creamos el grafo o no
			mlp.setNominalToBinaryFilter(true); 	// Para pasar variables nominales a binarias, de manera automática.
			mlp.setHiddenLayers("4"); 				/* Las capas ocultas creadas para la red.
			  											El valor debería ser una lista de números naturales separada por comas,
					  									o las letras 'a' = (atributos + clases) / 2, 
					  									'i' = atributos, 
					  									'o' = clases, 
					  									't' = atributos .+ clases (para comodines).
					  									Por defecto = a */
			mlp.setNormalizeNumericClass(true); 	// Indicar acá si las clases numéricas serán normalizadas o no.
			mlp.setNormalizeAttributes(true); 		// Indicar acá si los atributos serán normalizados o no.
			mlp.setReset(true); 					// Indicar acá si se permite que la red sea reiniciada o no.
			mlp.setDecay(false); 					// Indicar acá si se permite que la tasa de aprendizaje pueda decrecer o no.
			mlp.setDoNotCheckCapabilities(false); 	// Indicar acá si se permite que la librería pueda analizar las capacidades de la red o no.
			mlp.setDebug(false); 					// Indicar acá si la ejecución es depurable o no.

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
