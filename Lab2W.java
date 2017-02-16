import java.io.*;
import java.util.*;

// Stores info of the input files
// Number of Features, Feature Names, Feature Values and Label Values
class DataInfo{
	int numFeatures;
	String[] featureNames;
	List<List<String>> featureValues;
	String[] labelValues;

	public void set(int numFeatures, String[] featureNames, List<List<String>> featureValues, List<List<String>> labelValues){

	}

	// Print out DataInfo
	public void print(){
		System.out.println("Number of features: " + numFeatures);

		System.out.println("Features:");
		for(int i = 0; i < numFeatures; i++){
			System.out.print("\t" + featureNames[i] + "-> ");

			for(int j = 0; j < featureValues.get(i).size() - 1; j++){
				System.out.print(featureValues.get(i).get(j) + ", ");
			}

			System.out.println(featureValues.get(i).get(featureValues.get(i).size() - 1));
		}

		System.out.println("Labels: " + labelValues[0] + ", " + labelValues[1]);
	}

	// Return the index of the value for the feature[featureIndex], return -1 if not found
	public int getFeatureValueIndex(int featureIndex, String value){
		if(featureIndex >= numFeatures){
			return -1;
		}

		return featureValues.get(featureIndex).indexOf(value);
	}

	// Return the label index, return -1 if not found
	public int getLabelIndex(String label){
		if(label.equals(labelValues[0])){
			return 0;
		}
		else if(label.equals(labelValues[1])){
			return 1;
		}
		else{
			return -1;
		}
	}

	public String indexToLabel(int x){
		if(x < 0 || x >= 2){
			System.err.println("Label index out of bounds.");
			System.exit(-1);
		}

		return labelValues[x];
	}
}

// Scanner that ignore comments
class CommentScanner{
	private Scanner scn;
	private	String nextWord;
	private boolean nextExist;
	private String fileName;

	public CommentScanner(String fileName){
		this.fileName = fileName;
		File inFile = new File(fileName);
		try{
			scn = new Scanner(inFile).useDelimiter("#.*\n|//.*\n|\n");
		} catch (FileNotFoundException ex){
			System.err.println("Error: File " + fileName + " is not found.");
			System.exit(-1);
		}

		nextExist = true;
		setNext();
	}

	private void setNext(){
		while(scn.hasNext()){
			nextWord = scn.next().trim();

			if(!nextWord.equals("")){
				return;
			}
		}
		nextExist = false;
	}

	// Return true if there is a valid string in the buffer
	public boolean hasNext(){
		return nextExist;
	}

	// Return the next integer
	public int nextInt(){
		if(!nextExist){
			System.err.println("Scanner has no words left.");
			System.exit(-1);
		}

		int x = -1;

		try{
			x = Integer.parseInt(nextWord);
		} catch (NumberFormatException ex){
		}
		
		setNext();

		return x;
	}

	// Return the next string (which is a whole line)
	public String next(){
		if(!nextExist){
			System.err.println("Scanner has no words left.");
			System.exit(-1);
		}

		String x = nextWord;
		setNext();

		return x;
	}

	// Return the file name of the file this scanner is reading from
	public String getFileName(){
		return fileName;
	}
}

// Data structure to load and store examples
class ExampleList{
	// First item is the label while the rest are features
	int[][] examples;
	int numExamples;

	public ExampleList(CommentScanner scn, DataInfo di){
	}

	// Print out all examples
	public void print(){
		for(int i = 0; i < numExamples; i++){
			System.out.print(i + 1 + ": ");

			for(int j = 0; j < examples[0].length - 1; j++){
				System.out.print(examples[i][j] + ", ");
			}

			System.out.println(examples[i][examples[0].length - 1]);
		}
	}
}

class ProteinData{
	private List<List<String>> proteins;
	private List<List<String>> labels;
	private DataInfo di;

	public ProteinData(CommentScanner scn){
		proteins = new ArrayList<List<String>>();
		labels = new ArrayList<List<String>>();
		List<String> protein = new ArrayList<String>();
		List<String> label = new ArrayList<String>();
		Boolean reset = false;

		while(scn.hasNext()){
			String in = scn.next().trim().toLowerCase();

			if(in.equals("<>") || in.equals("<end>") || in.equals("end")){
				reset = true;
				continue;
			}

			if(reset){
				// New protein
				proteins.add(protein);
				labels.add(label);
				protein = new ArrayList<String>();
				label = new ArrayList<String>();
				reset = false;
			}

			// Same protein
			String[] inSplit = in.split(" ");
			protein.add(inSplit[0].trim().toLowerCase());
			label.add(inSplit[1].trim().toLowerCase());
		}

		proteins.add(protein);
		labels.add(label);
		proteins.remove(0);
		labels.remove(0);

		setDataInfo();
	}

	public void setDataInfo(){
		di = new DataInfo();
		Set<String> features = new HashSet<String>();

		for(List<String> list : proteins){
			for(String x : list){
				features.add(x);
			}
		}

		System.out.println(features.toString());
	}
}

// Single Perceptron
class Perceptron{
	DataInfo di;
	double weights[];
	double learningRate;

	public Perceptron(DataInfo di){
		new Perceptron(di, 0.1);
	}

	public Perceptron(DataInfo di, double learningRate){
		this.di = di;
		this.learningRate = learningRate;

		Random rdm = new Random();
		weights = new double[di.numFeatures + 1];

		for(int i = 0; i < di.numFeatures + 1; i++){
			weights[i] = rdm.nextDouble();
		}
	}

	// Get the weights of this perceptron
	public double[] getWeights(){
		return weights;
	}

	// Set the weights of this perceptron
	public void setWeights(double[] x){
		weights = x;
	}

	// Train the perceptron with examples
	public void train(ExampleList ex){
		for(int i = 0; i < ex.numExamples; i++){
			double diff = learningRate * (ex.examples[i][weights.length - 1] - predict(ex.examples[i]));

			for(int j = 0; j < weights.length - 1; j++){
				weights[j] += diff * ex.examples[i][j];
			}

			weights[weights.length - 1] += diff;
		}
	}

	// Predict the value of an example
	public int predict(int[] row){
		double sum = 0;

		for(int i = 0; i < row.length; i++){
			sum += row[i] * weights[i];
		}

		sum += weights[weights.length - 1];

		if(sum >= 0){
			return 1;
		}
		else{
			return 0;
		}
	}

	// Prediction on an entire set of examples
	public double test(ExampleList ex){
		double sum = 0;

		for(int i = 0; i < ex.numExamples; i++){
			if(predict(ex.examples[i]) == ex.examples[i][ex.examples[0].length - 1]){
				sum += 1;
			}
		}

		return sum / ex.numExamples;
	}

	// Prediction on an entire set of examples with output
	public double testWithOutput(ExampleList ex){
		double sum = 0;

		for(int i = 0; i < ex.numExamples; i++){
			int prediction = predict(ex.examples[i]);
			if(prediction == ex.examples[i][ex.examples[0].length - 1]){
				sum += 1;
			}
			System.out.println(di.indexToLabel(prediction));
		}
		
		return sum / ex.numExamples;	
	}

	// Set the learning rate of the perceptron
	public void setLearningRate(double x){
		learningRate = x;
	}
}	

public class Lab2W{
	// Check for correct program arguments
	public static void checkArgs(String[] args){
		if(args.length != 1){
			System.err.println("Usage: Lab2W <fileNameOfData>");
			System.exit(-1);
		}
	}

	public static void main(String[] args){
		checkArgs(args);

		// Scanner that ignore comments for files
		CommentScanner inputScn = new CommentScanner(args[0]);

		ProteinData proteinData = new ProteinData(inputScn);
		
		// Load examples
		// ExampleList trainEx = new ExampleList(inputScn, dataInfo);

		// Initialize perceptrons
		// Perceptron perceptron = new Perceptron(dataInfo, 0.1);

		// Train y=numPerceptrons and pick the best one
		// Use early stopping for each perceptron
		// Early stopping rule: Stop if accuracy does not increase after x=patience epoch
		// int numPerceptrons = 50;
		// int patience = 30;

		// for(int i = 0; i < numPerceptrons; i++){
		// 	Perceptron perceptronNew = new Perceptron(dataInfo, 0.1);

		// 	double acc = perceptronNew.test(tuneEx);
		// 	double weights[] = perceptronNew.getWeights();

		// 	for(int j = 0; j < patience; j++){
		// 		perceptronNew.train(trainEx);
		// 		double newAcc = perceptronNew.test(tuneEx);

		// 		if(newAcc > acc){
		// 			acc = newAcc;
		// 			j = 0;
		// 			weights = perceptronNew.getWeights();
		// 		}

		// 		perceptronNew.setWeights(weights);
		// 	}

		// 	if(perceptronNew.test(tuneEx) > perceptron.test(tuneEx)){
		// 		perceptron = perceptronNew;
		// 	}
		// }

		// Print out result and overall accuracy
		// System.out.printf("Overall Accuracy: %.2f\n", perceptron.testWithOutput(testEx) * 100);
	}
}