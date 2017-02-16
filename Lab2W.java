import java.io.*;
import java.util.*;

// Stores info of the input files
// Number of Features, Feature Names, Feature Values and Label Values
class DataInfo{
	// This case assumes that there is only one feature and one label
	int numFeatures; // Number of values that the feature can take
	int numLabels;
	List<String> featureValues;
	List<String> labelValues;

	public DataInfo(){
		this(0, 0, null, null);
	}

	public DataInfo(int numFeatures, int numLabels, List<String> featureValues, List<String> labelValues){
		set(numFeatures, numLabels, featureValues, labelValues);
	}

	public void set(int numFeatures, int numLabels, List<String> featureValues, List<String> labelValues){
		this.numFeatures = numFeatures;
		this.numLabels = numLabels;
		this.featureValues = featureValues;
		this.labelValues = labelValues;
	}

	// Print out DataInfo
	public void print(){
		System.out.println("Number of features: " + numFeatures);
		System.out.println("Number of labels: " + numLabels);

		System.out.println("Features:");
		System.out.println(featureValues.toString());

		System.out.println("Labels:");
		System.out.println(labelValues.toString());		
	}

	// Return the index of the value for the feature[featureIndex], return -1 if not found
	public int getFeatureValueIndex(String value){
		return featureValues.indexOf(value);
	}

	// Return the label index, return -1 if not found
	public int getLabelValueIndex(String label){
		return labelValues.indexOf(label);
	}

	public String indexToLabel(int x){
		if(x < 0 || x > numLabels - 1){
			System.err.println("Label index out of bounds.");
			System.exit(-1);
		}

		return labelValues.get(x);
	}

	public String indexToFeature(int x){
		if(x < 0 || x > numFeatures){
			System.err.println("Label index out of bounds.");
			System.exit(-1);
		}

		if(x == numFeatures){
			// Return padding as "NULL"
			return "NULL";
		}

		return featureValues.get(x);
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
	List<List<Integer>> examples;
	int numExamples;

	public ExampleList(){
		examples = new ArrayList<List<Integer>>();
		numExamples = 0;
	}

	// Set examples using sliding window of size windowSize
	public void setExamples(List<List<String>> features, List<List<String>> labels, int windowSize, DataInfo di){
		// Add examples
		for(List<String> protein : features){
			List<Integer> window = new ArrayList<Integer>();

			// The first window
			for(int i = 0; i < windowSize; i++){
				window.add(di.getFeatureValueIndex(protein.get(i)));
			}
			examples.add(new ArrayList<Integer>(window));

			// The rest of the windows
			for(int i = 1; i < protein.size() - windowSize + 1; i++){
				window.remove(0);
				window.add(di.getFeatureValueIndex(protein.get(i + windowSize - 1)));
				examples.add(new ArrayList<Integer>(window));
			}
		}

		// Add labels
		int k = 0;
		for(List<String> label : labels){
			for(String l : label){
				examples.get(k).add(0, di.getLabelValueIndex(l));
				k++;
			}
		}

		numExamples = examples.size();
	}

	// Print out all examples
	public void print(){
		int i = 0;
		for(List<Integer> x : examples){
			System.out.print("Example " + i + ": ");
			i++;
			System.out.println(x.toString());
		}
	}
}

class ProteinData{
	private List<List<String>> proteins;
	private List<List<String>> proteinTypes;
	DataInfo di;
	ExampleList trainList;
	ExampleList tuneList;
	ExampleList testList;

	public ProteinData(CommentScanner scn){
		proteins = new ArrayList<List<String>>();
		proteinTypes = new ArrayList<List<String>>();
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
				proteinTypes.add(label);
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
		proteinTypes.add(label);
		proteins.remove(0);
		proteinTypes.remove(0);

		addPadding(proteins, 8);

		// Set the feature index, 1 of N encoding
		setDataInfo();
		setExample();
	}

	// Add padding of pSize to each protein
	public void addPadding(List<List<String>> proteins, int pSize){
		for(List<String> protein : proteins){
			for(int i = 0; i < pSize; i++){
				protein.add("PADDING_AMINO_ACID");
				protein.add(0, "PADDING_AMINO_ACID");
			}
		}
	}

	// Split the data into Train, Tune and Test set based on instructions from slides
	public void setExample(){
		List<Integer> trainIdx = new ArrayList<Integer>();
		List<Integer> tuneIdx = new ArrayList<Integer>();
		List<Integer> testIdx = new ArrayList<Integer>();

		// Base on slides
		for(int i = 0; i < proteins.size(); i++){
			trainIdx.add(i);
		}

		int j = 4;
		while(j < proteins.size()){
			tuneIdx.add(j);
			j += 5;
		}

		j = 5;
		while(j < proteins.size()){
			testIdx.add(j);
			j += 5;
		}

		for(int i : tuneIdx){
			trainIdx.remove(new Integer(i));
		}
		for(int i : testIdx){
			trainIdx.remove(new Integer(i));
		}

		trainList = new ExampleList();
		setExampleList(trainList, trainIdx);

		tuneList = new ExampleList();
		setExampleList(tuneList, tuneIdx);

		testList = new ExampleList();
		setExampleList(testList, testIdx);
	}

	// set the examplelist el to contain proteins of index in indexes
	public void setExampleList(ExampleList el, List<Integer> indexes){
		List<List<String>> tempFeatureList = new ArrayList<List<String>>();
		List<List<String>> tempLabelList = new ArrayList<List<String>>();

		for(int i : indexes){
			tempFeatureList.add(proteins.get(i));
			tempLabelList.add(proteinTypes.get(i));
		}

		el.setExamples(tempFeatureList, tempLabelList, 17, di);
	}

	// Set the dataInfo which stores information to encode features and labels to integers
	public void setDataInfo(){
		Set<String> features = new HashSet<String>();
		Set<String> labels = new HashSet<String>();

		for(List<String> list : proteins){
			for(String x : list){
				features.add(x);
			}
		}

		for(List<String> list : proteinTypes){
			for(String x : list){
				labels.add(x);
			}
		}

		di = new DataInfo(features.size(), labels.size(), new ArrayList<String>(features), new ArrayList<String>(labels));
	}
}

// Single Perceptron
// class Perceptron{
// 	DataInfo di;
// 	double weights[];
// 	double learningRate;

// 	public Perceptron(DataInfo di){
// 		new Perceptron(di, 0.1);
// 	}

// 	public Perceptron(DataInfo di, double learningRate){
// 		this.di = di;
// 		this.learningRate = learningRate;

// 		Random rdm = new Random();
// 		weights = new double[di.numFeatures + 1];

// 		for(int i = 0; i < di.numFeatures + 1; i++){
// 			weights[i] = rdm.nextDouble();
// 		}
// 	}

// 	// Get the weights of this perceptron
// 	public double[] getWeights(){
// 		return weights;
// 	}

// 	// Set the weights of this perceptron
// 	public void setWeights(double[] x){
// 		weights = x;
// 	}

// 	// Train the perceptron with examples
// 	public void train(ExampleList ex){
// 		for(int i = 0; i < ex.numExamples; i++){
// 			double diff = learningRate * (ex.examples[i][weights.length - 1] - predict(ex.examples[i]));

// 			for(int j = 0; j < weights.length - 1; j++){
// 				weights[j] += diff * ex.examples[i][j];
// 			}

// 			weights[weights.length - 1] += diff;
// 		}
// 	}

// 	// Predict the value of an example
// 	public int predict(int[] row){
// 		double sum = 0;

// 		for(int i = 0; i < row.length; i++){
// 			sum += row[i] * weights[i];
// 		}

// 		sum += weights[weights.length - 1];

// 		if(sum >= 0){
// 			return 1;
// 		}
// 		else{
// 			return 0;
// 		}
// 	}

// 	// Prediction on an entire set of examples
// 	public double test(ExampleList ex){
// 		double sum = 0;

// 		for(int i = 0; i < ex.numExamples; i++){
// 			if(predict(ex.examples[i]) == ex.examples[i][ex.examples[0].length - 1]){
// 				sum += 1;
// 			}
// 		}

// 		return sum / ex.numExamples;
// 	}

// 	// Prediction on an entire set of examples with output
// 	public double testWithOutput(ExampleList ex){
// 		double sum = 0;

// 		for(int i = 0; i < ex.numExamples; i++){
// 			int prediction = predict(ex.examples[i]);
// 			if(prediction == ex.examples[i][ex.examples[0].length - 1]){
// 				sum += 1;
// 			}
// 			// System.out.println(di.indexToLabel(prediction));
// 		}
		
// 		return sum / ex.numExamples;	
// 	}

// 	// Set the learning rate of the perceptron
// 	public void setLearningRate(double x){
// 		learningRate = x;
// 	}
// }	

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