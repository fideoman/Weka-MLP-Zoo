package cl.usach.rn.mlp;

import java.io.File;
import java.io.FileReader;
import java.util.Arrays;

import cl.usach.rn.mlp.utilidades.IWSS;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

@SuppressWarnings("unused")
public class AnalisisZoologicoUCI {

	public static void main(String[] args) {
		AnalisisZoologicoUCI.pruebaMLP("zoo.arff", 0.1, 0.1, 1000, 0, 0, 20, false, false, true, "3", true, true, true, false, false, false);
	}
	
	public static void pruebaMLP(String archivo, double learningRate, double momentum, int tiempoEntrenamiento, int tamanoValidacion, int semilla, int maximoErroresVal, boolean gui, boolean autoBuild,
			boolean filtroNominalBinario, String capasOcultas, boolean normalizarClasesNumericas, boolean normalizarAtributos, boolean admiteReseteo, boolean admiteDecaimiento, boolean noRevisarCarac, boolean debug) {
		ClassLoader cargadorContexto = Thread.currentThread().getContextClassLoader();

		try {
			// Lectura del set de entrenamiento y prueba. Se cierran los archivos inmediatamente, una vez leídas las instancias.
			FileReader lectorEntrenamiento = new FileReader(new File(cargadorContexto.getResource(archivo).toURI()));
			FileReader lectorPrueba = new FileReader(new File(cargadorContexto.getResource(archivo).toURI()));
			Instances entrenamiento = new Instances(lectorEntrenamiento);
			Instances prueba = new Instances(lectorPrueba);
			lectorEntrenamiento.close();
			lectorPrueba.close();
			
			// Establecemos la columna que tiene la clase. Ese es el "target".
			entrenamiento.setClassIndex(entrenamiento.numAttributes() - 1);
			prueba.setClassIndex(prueba.numAttributes() - 1);

			// Clasificador: Perceptrón multicapa.
			MultilayerPerceptron mlp = new MultilayerPerceptron();
			
			// Rutina de ranking de atributos para eliminación.
			/* Implementado por Pablo Bermejo, 
			 * Jose A. Gomez and Jose,(2011). Improving
			 * Incremental Wrapper-Based Subset Selection via Replacement and Early
			 * Stopping. International Journal of Pattern Recognition and Artificial
			 * Intelligence. 25(5):605-625 */
			
			// Entrega, por cada atributo, su nivel de "eliminable".
			// IWSS iwss = new IWSS();
			// System.out.println(Arrays.toString(iwss.getRanking(prueba)));
			// Resultado del Zoo Dataset original:
			// [0, 4, 3, 8, 1, 2, 9, 10, 14, 13, 19, 15, 12, 17, 5, 6, 21, 11, 18, 7, 16, 20]
			
			// En concreto, los 5 más eliminables son:
			// 5_patas: 21, pequeño: 20, respirador: 19, 8_patas: 18, 0_patas: 17

			// Test suite
			
			// Parámetros:
			mlp.setLearningRate(learningRate); 						// Tasa de aprendizaje. De 0 a 1. Por defecto: 0.3
			mlp.setMomentum(momentum); 								// Momentum para el algoritmo de retro-propagación. De 0 a 1. Por defecto: 0.2
			mlp.setTrainingTime(tiempoEntrenamiento); 				// Tiempo de entrenamiento, en épocas. Por defecto: 500
			mlp.setValidationSetSize(tamanoValidacion); 			/* Tamaño (en porcentaje) del grupo de validación 
																		usado para terminar el entrenamiento. 
																		Si no es cero, puede pre-calcular número de épocas. 
																		De 0 a 100. Por defecto: 0 */
			mlp.setSeed(semilla); 									// Semilla. El valor para alimentar al generator de números aleatorios. Por defecto: 0
			mlp.setValidationThreshold(maximoErroresVal); 			/* El número máximo de errores consecutivos permitido para las pruebas de validación
																		antes que la red termine. Valor debe ser mayor que cero. Por defecto: 20*/
			mlp.setGUI(gui); 										// Para abrir, o no, una guía gráfica.
			mlp.setAutoBuild(autoBuild); 							// En el caso de elegir GUI, ésta opción indica si creamos el grafo o no
			mlp.setNominalToBinaryFilter(filtroNominalBinario); 	// Para pasar variables nominales a binarias, de manera automática.
			mlp.setHiddenLayers(capasOcultas); 						/* Las capas ocultas creadas para la red.
			  															El valor debería ser una lista de números naturales separada por comas,
					  													o las letras 'a' = (atributos + clases) / 2, 
					  													'i' = atributos, 
					  													'o' = clases, 
					  													't' = atributos .+ clases (para comodines).
					  												Por defecto = a */
			mlp.setNormalizeNumericClass(normalizarClasesNumericas); // Indicar acá si las clases numéricas serán normalizadas o no.
			mlp.setNormalizeAttributes(normalizarAtributos); 		// Indicar acá si los atributos serán normalizados o no.
			mlp.setReset(admiteReseteo); 							// Indicar acá si se permite que la red sea reiniciada o no.
			mlp.setDecay(admiteDecaimiento); 						// Indicar acá si se permite que la tasa de aprendizaje pueda decrecer o no.
			mlp.setDoNotCheckCapabilities(noRevisarCarac); 			// Indicar acá si se permite que la librería pueda analizar las capacidades de la red o no.
			mlp.setDebug(debug); 									// Indicar acá si la ejecución es depurable o no.

			// Se construye el clasificador
			mlp.buildClassifier(entrenamiento);

			// Se evalúa el modelo con el grupo de prueba
			Evaluation eval = new Evaluation(entrenamiento);
			eval.evaluateModel(mlp, prueba);
			
			// Imprimir todos los resultados.		
			System.out.println("Tasa de aprendizaje:" + mlp.getLearningRate());
			System.out.println("Momentum: " + mlp.getMomentum());
			System.out.println("Tiempo de entrenamiento: " + mlp.getTrainingTime());
			System.out.println("Tamaño de validación: " + mlp.getValidationSetSize());
			System.out.println("Semilla: " + mlp.getSeed());
			System.out.println("Errores de validacion aceptados :" + mlp.getValidationThreshold());
			System.out.println("Tiene filtro de nominal a binario?: " + mlp.getNominalToBinaryFilter());
			System.out.println("Capas ocultas: " + mlp.getHiddenLayers());
			System.out.println("Se normalizan las clases numéricas?: " + mlp.getNormalizeNumericClass());
			System.out.println("Se normalizan los atributos?: " + mlp.getNormalizeAttributes());
			System.out.println("Puede resetearse la red internamente?: " + mlp.getReset());
			System.out.println("Puede la red tener un aprendizaje en decaimiento?: " + mlp.getDecay());
			System.out.println("Chequeo características?: " + mlp.getDoNotCheckCapabilities());
			System.out.println("Estoy en debugging?: " + mlp.getDebug());
			System.out.println("====================");
			System.out.println("Error medio absoluto: " + eval.meanAbsoluteError());
			System.out.println("Error medio cuadrado: " + eval.rootMeanSquaredError());
			System.out.println("Error relativo absoluto: " + eval.relativeAbsoluteError()+" %");
			System.out.println("Error relativo cuadrado: " + eval.rootRelativeSquaredError()+" %");
			System.out.println("Número de instancias: " + eval.numInstances());
			System.out.println("Costo de las clasificaciones incorrectas sobre el total (promedio): " + eval.avgCost());
			System.out.println("Costo total SUM (costo de cada predicción x ancho de cada instancia) : " + eval.totalCost());
			System.out.println("Instancias correctamente clasificadas (o suma de sus pesos): " + eval.correct());
			System.out.println("Porcentaje correcto: " + eval.pctCorrect());
			System.out.println("Instancias incorrectamente clasificadas (o suma de sus pesos): " + eval.incorrect());
			System.out.println("Porcentaje incorrecto: " + eval.pctIncorrect());
			System.out.println("Instancias no clasificadas (o suma de sus pesos): " + eval.unclassified());
			System.out.println("Porcentaje no clasificado: " + eval.pctUnclassified());
			System.out.println("Tasa de error: " + eval.errorRate());
			System.out.println("Estadística Kappa (concordancia): " + eval.kappa());
			System.out.println("Puntaje medio Kononenko & Bratko: " + eval.KBMeanInformation());
			System.out.println("Entropía total del esquema: " + eval.SFSchemeEntropy());
			System.out.println("Tamaño promedio de las regiones predecidas: " + eval.sizeOfPredictedRegions());
			System.out.println(eval.toClassDetailsString("Detalle de clases:"));
			System.out.println(eval.toMatrixString("Matriz de confusión:"));			
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}
}
