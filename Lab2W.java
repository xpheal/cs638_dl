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
	public double getFeatureValueIndex(String value){
		return featureValues.indexOf(value);
	}

	// Return the label index, return -1 if not found
	public double getLabelValueIndex(String label){
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
	List<List<Double>> examples;
	int numExamples;
	int windowSize;

	public ExampleList(){
		examples = new ArrayList<List<Double>>();
		numExamples = 0;
		windowSize = 1;
	}

	// Set examples using sliding window of size windowSize
	public void setExamples(List<List<String>> features, List<List<String>> labels, int windowSize, DataInfo di){
		// Add examples
		for(List<String> protein : features){
			List<Double> window = new ArrayList<Double>();

			// The first window
			for(int i = 0; i < windowSize; i++){
				window.add(di.getFeatureValueIndex(protein.get(i)));
			}
			examples.add(new ArrayList<Double>(window));

			// The rest of the windows
			for(int i = 1; i < protein.size() - windowSize + 1; i++){
				window.remove(0);
				window.add(di.getFeatureValueIndex(protein.get(i + windowSize - 1)));
				examples.add(new ArrayList<Double>(window));
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
		this.windowSize = windowSize;
	}

	// Print out all examples
	public void print(){
		int i = 0;
		for(List<Double> x : examples){
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
		trainIdx.add(0);
		trainIdx.add(1);
		trainIdx.add(2);
		// for(int i = 0; i < proteins.size(); i++){
		// 	trainIdx.add(i);
		// }

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
class Perceptron{
	String actFunc;
	int numIn;
	List<Double> inputs;
	double weights[];
	double learningRate;
	double doutdnet;

	// actFunc = activation Function of the perceptron, can be either "rec" for rectified linear or "sig" for sigmoidal
	// numIn = number of input weights
	public Perceptron(int numIn, String actFunc, double learningRate){
		if(!actFunc.equals("rec") && !actFunc.equals("sig")){
			System.err.println("Invalid activation function parameter");
			System.exit(-1);
		}

		this.actFunc = actFunc;
		this.numIn = numIn;
		this.learningRate = learningRate;
		this.doutdnet = 0;
		this.inputs = null;

		weights = new double[numIn + 1]; // +1 for bias

		// Initialize weights
		for(int i = 0; i < weights.length; i++){
			weights[i] = Math.random() * 2 - 1;
		}
	}

	public double feedForward(List<Double> inputs){
		this.inputs = inputs;

		if(inputs.size() != numIn){
			System.err.println("Wrong number of inputs for this perceptron!");
			System.exit(-1);
		}

		double net = 0;

		for(int i = 0; i < numIn; i++){
			net += inputs.get(i) * weights[i];
		}

		net += weights[numIn]; // bias

		switch(actFunc){
			case "rec":
				return recL(net);
			case "sig":
				return sigM(net);
			default:
				System.err.println("Error, shouldn't reach this line of code");
				System.exit(-1);
				return -1;
		}
	}

	// Delta equals to (dError/dout)
	// Returns newDeltai = (dError/dout * dout/dnet) * wi for backpropagation
	public double[] backPropagate(double delta){
		double newDelta = delta * doutdnet;
		double deltaList[] = new double[numIn];

		// Update all weights
		for(int i = 0; i < numIn; i++){
			deltaList[i] = newDelta * weights[i];
			double ll = learningRate * newDelta * inputs.get(i);
			weights[i] -= ll;
		}

		// Update bias
		weights[numIn] -= learningRate * newDelta;

		return deltaList;
	}
	
	private double sigM(double x){
		double out = 1 / (1 + Math.exp(-x));
		doutdnet = out * (1 - out);
		return out;
	}

	private double recL(double x){
		if(x >= 0){
			doutdnet = 1;
			return x;
		}
		else{
			doutdnet = 0;
			return 0;
		}
	}

	public double[] getWeights(){
		return weights;
	}
}

class NeuralNetwork{
	int numHiddenUnits;
	int numClass;
	List<Perceptron> hiddenLayer;
	List<Double> hiddenLayerOutputs;
	List<Perceptron> outputLayer;
	double outputs[];

	// numInputs = number of inputs into the neural network
	// numHiddenUnits = number of Perceptrons in the hidden layer
	// numClass = number of classes to classify or number of distinct label values
	public NeuralNetwork(int numInputs, int numHiddenUnits, int numClass){
		this.numHiddenUnits = numHiddenUnits;
		this.numClass = numClass;
		hiddenLayerOutputs = new ArrayList<Double>();
		hiddenLayer = new ArrayList<Perceptron>();

		for(int i = 0; i < numHiddenUnits; i++){
			hiddenLayer.add(new Perceptron(numInputs, "sig", 0.01));
			hiddenLayerOutputs.add(0.0);
		}

		outputLayer = new ArrayList<Perceptron>();
		outputs = new double[numClass];

		for(int i = 0; i < numClass; i++){
			outputLayer.add(new Perceptron(numHiddenUnits, "sig", 0.01));
		}
	}

	public void train(List<List<Double>> examples){
		for(List<Double> example : examples){
			// p.backPropagate(-(example.get(0) - predict(example.subList(1, example.size()))));
			double label = example.get(0);
			predict(example.subList(1, example.size())); // Feedforward

			double actuals[] = new double[numClass];
			actuals[(int)label] = 1;

			List<double[]> deltas = new ArrayList<double[]>();

			for(int i = 0; i < numClass; i++){
				deltas.add(outputLayer.get(i).backPropagate(outputs[i] - actuals[i]));
			}

			for(int i = 0; i < numHiddenUnits; i++){
				double delta = 0;

				for(int j = 0; j < numClass; j++){
					delta += deltas.get(j)[i];
				}

				hiddenLayer.get(i).backPropagate(delta);
			}
		}
	}

	public double predict(List<Double> inputs){
		for(int i = 0; i < numHiddenUnits; i++){
			hiddenLayerOutputs.set(i, hiddenLayer.get(i).feedForward(inputs));
		}

		for(int i = 0; i < numClass; i++){
			outputs[i] = outputLayer.get(i).feedForward(hiddenLayerOutputs);
		}

		return classify(outputs);
	}

	// Assumes that output is of at least length 2, binary classification
	public double classify(double[] outputs){
		int idx = 0;
		double value = outputs[0];

		for(int i = 1; i < outputs.length; i++){
			if(outputs[i] > value){
				value = outputs[i];
				idx = i;
			}
		}

		return idx;
	}

	// Return the accuracy of the test
	public double test(List<List<Double>> examples){
		return test(examples, false);
	}

	// Return the accuracy of the test, print out results if debug is True
	public double test(List<List<Double>> examples, Boolean debug){
		double numCorrect = 0;

		for(List<Double> example : examples){
			double label = example.get(0);
			double output = predict(example.subList(1, example.size()));
			
			if(output == label){
				numCorrect++;
			}

			if(debug){
				System.out.println(output + " : " + label);
			}
		}

		return numCorrect / examples.size();
	}

	// Print out the weights of each perceptron
	public void debugWeights(){
		System.out.println("HiddenLayer:");

		for(Perceptron p : hiddenLayer){
			System.out.println(Arrays.toString(p.getWeights()));
		}

		System.out.println("OutputLayer:");

		for(Perceptron p : outputLayer){
			System.out.println(Arrays.toString(p.getWeights()));
		}
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

		NeuralNetwork nn = new NeuralNetwork(proteinData.trainList.windowSize, 17, proteinData.di.numLabels);

		// proteinData.di.print();
		System.out.println(nn.test(proteinData.trainList.examples));

		for(int i = 0; i < 100; i++){
			nn.train(proteinData.trainList.examples);
		}
		System.out.println(nn.test(proteinData.trainList.examples, true));

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