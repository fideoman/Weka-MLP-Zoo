package cl.usach.rn.mlp;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import cl.usach.rn.mlp.utilidades.IWSS;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class AnalisisZoologicoUCI {

	public static ArrayList<Double> erroresMediaAbsoluta = new ArrayList<Double>();
	public static ArrayList<Double> erroresMediaRaizCuadrada = new ArrayList<Double>();
	public static ArrayList<Double> erroresRelativoAbsoluto = new ArrayList<Double>();
	public static ArrayList<Double> erroresRelativoRaizCuadrada = new ArrayList<Double>();
	public static ArrayList<Double> costosPromedioIncorrectas = new ArrayList<Double>();
	public static ArrayList<Double> porcentajesCorrecto = new ArrayList<Double>();
	public static ArrayList<Double> tasasError = new ArrayList<Double>();
	public static ArrayList<Double> concordanciasKappa = new ArrayList<Double>();		
	
	public static void main(String[] args) throws FileNotFoundException, IOException {
		// Timestamp de la ejecución
		long timeStamp = (new Timestamp(System.currentTimeMillis())).getTime();		
		
		// TODO: Bien fea la redirección, pero en honor del tiempo... Todos los System.out serán depositados en dicho archivo. Se puede cambiar.
		System.setOut(new PrintStream(new File(System.getProperty("user.dir")+"/output/salidaEjecucion-"+ timeStamp +".txt")));
		
		int z = 1;
		
		// Listas de parámetros
		int[] numeroAttribs = new int[]{ 0, 5 };
		double[] tasasAprendizaje = new double[] {0.1, 0.3};
		double[] momentums = new double[] {0.1, 0.3};
		int[] tamanoValidaciones = new int[] {0, 20};
		int[] tiempoEntrenamiento = new int[] {3000, 8000};
		int[] maximoErrores = new int[] {10, 20};
		String[] capasOcultas = new String[] {"2", "3"}; 
				
		for(int i = 0; i <= 127; i++) {
			char[] combinacion = String.format("%7s", Integer.toBinaryString(i)).replace(' ', '0').toCharArray();
			AnalisisZoologicoUCI.pruebaMLP(z++, "zoo.arff", numeroAttribs[Character.getNumericValue(combinacion[6])], tasasAprendizaje[Character.getNumericValue(combinacion[5])], momentums[Character.getNumericValue(combinacion[4])], tamanoValidaciones[Character.getNumericValue(combinacion[3])], tiempoEntrenamiento[Character.getNumericValue(combinacion[2])], maximoErrores[Character.getNumericValue(combinacion[1])], capasOcultas[Character.getNumericValue(combinacion[0])]);
		}	
		
		capasOcultas = new String[] {"4", "4,4"};
				
		for(int i = 0; i <= 127; i++) {
			char[] combinacion = String.format("%7s", Integer.toBinaryString(i)).replace(' ', '0').toCharArray();
			AnalisisZoologicoUCI.pruebaMLP(z++, "zoo.arff", numeroAttribs[Character.getNumericValue(combinacion[6])], tasasAprendizaje[Character.getNumericValue(combinacion[5])], momentums[Character.getNumericValue(combinacion[4])], tamanoValidaciones[Character.getNumericValue(combinacion[3])], tiempoEntrenamiento[Character.getNumericValue(combinacion[2])], maximoErrores[Character.getNumericValue(combinacion[1])], capasOcultas[Character.getNumericValue(combinacion[0])]);
		}	
		
		grafico("erroresMediaAbsoluta"+timeStamp, "Errores Medias Absolutas", "Error Media Absoluta", erroresMediaAbsoluta);
		grafico("erroresMediaRaizCuadrada"+timeStamp, "Errores Raices Medias Cuadradas", "Error Raíz Media Cuadrada", erroresMediaRaizCuadrada);
		grafico("erroresRelativoAbsoluto"+timeStamp, "Errores Relativos Absolutos", "Error Relativo Absoluta", erroresRelativoAbsoluto);
		grafico("erroresRelativoRaizCuadrada"+timeStamp, "Errores Raíces Relativas Cuadradas", "Error Raíz Relativa Cuadrada", erroresRelativoRaizCuadrada);
		grafico("costosPromedioIncorrectas"+timeStamp, "Costos Promedio de Incorrectas", "Costo Promedio Incorrecto", costosPromedioIncorrectas);
		grafico("porcentajesCorrecto"+timeStamp, "Porcentajes Correctos", "Porcentaje Correcto", porcentajesCorrecto);
		grafico("tasasError"+timeStamp, "Tasas de Error", "Tasa de Error", tasasError);
		grafico("concordanciasKappa"+timeStamp, "Niveles de Concordancia Kappa", "Nivel de Concordancia Kappa", concordanciasKappa);
	}
	
	public static void pruebaMLP(int id, String archivo, int numeroAttribsEliminar, double learningRate, double momentum, int tamanoValidacion, int tiempoEntrenamiento, int maximoErroresVal, String capasOcultas) throws FileNotFoundException {
		pruebaMLP(id, archivo, numeroAttribsEliminar, learningRate, momentum, tamanoValidacion, tiempoEntrenamiento, maximoErroresVal, capasOcultas, 
				0, false, false, true, true, true, true, false, false, false);
	}
	
	public static void pruebaMLP(int id, String archivo, int numeroAttribsEliminar, double learningRate, double momentum, int tamanoValidacion, int tiempoEntrenamiento, int maximoErroresVal, String capasOcultas, 
			int semilla, boolean gui, boolean autoBuild, boolean filtroNominalBinario,  boolean normalizarClasesNumericas, boolean normalizarAtributos, boolean admiteReseteo, 
			boolean admiteDecaimiento, boolean noRevisarCarac, boolean debug) throws FileNotFoundException {
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
			IWSS iwss = new IWSS();
			int[] listaEliminar = new int[numeroAttribsEliminar];
			List<Integer> listaRanking = Arrays.stream(iwss.getRanking(prueba)).boxed().collect(Collectors.toList());
			// Creamos un arreglo de índices que serán eliminados. Los más eliminables.
			int j = 0;
			for (int i = listaRanking.size(); i > (listaRanking.size() - numeroAttribsEliminar); i--) {
				listaEliminar[j] = listaRanking.indexOf(i-1);
				j++;
			}
			// Eliminamos
			Remove removerInstancias = new Remove();
			removerInstancias.setAttributeIndicesArray(listaEliminar);
			removerInstancias.setInputFormat(prueba);
			entrenamiento = Filter.useFilter(prueba, removerInstancias);
			prueba = Filter.useFilter(prueba, removerInstancias);
						
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
					  													't' = atributos + clases.
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
			System.out.println("ID de ejecución: " + id);
			System.out.println("Tasa de aprendizaje: " + mlp.getLearningRate());
			System.out.println("Momentum: " + mlp.getMomentum());
			System.out.println("Tiempo de entrenamiento máximo: " + mlp.getTrainingTime());
			System.out.println("Tamaño de validación: " + (mlp.getValidationSetSize() == 0 ? "0 (Se usa tiempo de Entrenamiento)" : mlp.getValidationSetSize()));
			System.out.println("Semilla: " + mlp.getSeed());
			System.out.println("Errores de validacion aceptados: " + mlp.getValidationThreshold());
			System.out.println("Tiene filtro de nominal a binario?: " + mlp.getNominalToBinaryFilter());
			System.out.println("Capas ocultas: " + mlp.getHiddenLayers());
			System.out.println("Se normalizan las clases numéricas?: " + mlp.getNormalizeNumericClass());
			System.out.println("Se normalizan los atributos?: " + mlp.getNormalizeAttributes());
			System.out.println("Puede resetearse la red internamente?: " + mlp.getReset());
			System.out.println("Puede la red tener un aprendizaje en decaimiento?: " + mlp.getDecay());
			System.out.println("Revisar características?: " + !mlp.getDoNotCheckCapabilities());
			System.out.println("Estoy en debugging?: " + mlp.getDebug());
			System.out.println("====================");
			System.out.println("Error media absoluta: " + eval.meanAbsoluteError());
			System.out.println("Error media raíz cuadrada: " + eval.rootMeanSquaredError());
			System.out.println("Error relativo absoluto: " + eval.relativeAbsoluteError()+" %");
			System.out.println("Error relativo raíz cuadrada: " + eval.rootRelativeSquaredError()+" %");
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
			
			// Guardar en listas globales
			erroresMediaAbsoluta.add(eval.meanAbsoluteError());
			erroresMediaRaizCuadrada.add(eval.rootMeanSquaredError());
			erroresRelativoAbsoluto.add(eval.relativeAbsoluteError());
			erroresRelativoRaizCuadrada.add(eval.rootRelativeSquaredError());
			costosPromedioIncorrectas.add(eval.avgCost());
			porcentajesCorrecto.add(eval.pctCorrect());
			tasasError.add(eval.errorRate());
			concordanciasKappa.add(eval.kappa());			
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}
	
	public static void grafico(String nombreArchivo, String titulo, String linea, ArrayList<Double> indice) throws FileNotFoundException, IOException {
		String rutaGrafico = System.getProperty("user.dir")+"/output/";
		XYSeries serie1 = new XYSeries(linea);
		for(int i=1;i<=indice.size();i++) {
			serie1.add(i, indice.get(i-1));
		}
		XYSeriesCollection datos = new XYSeriesCollection();
		datos.addSeries(serie1);
		JFreeChart chart = ChartFactory.createXYLineChart(titulo, "ID Ejecución", "Valor Indice", datos,
				PlotOrientation.VERTICAL, true, true, false);

		ChartUtilities.writeChartAsPNG(new FileOutputStream(rutaGrafico+"/"+ nombreArchivo+".png"), chart, 1280, 720);
	}
	
}
