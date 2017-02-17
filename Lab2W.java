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
		if(x < 0 || x > numFeatures - 1){
			System.err.println("Label index out of bounds.");
			System.exit(-1);
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
	int exampleSize;

	public ExampleList(){
		examples = new ArrayList<List<Double>>();
		numExamples = 0;
		windowSize = 1;
		exampleSize = 1;
	}

	// Set examples using sliding window of size windowSize
	public void setExamples(List<List<String>> features, List<List<String>> labels, int windowSize, DataInfo di){
		// Add examples
		List<List<Double>> encoding = generateEncoding(di.numFeatures);

		for(List<String> protein : features){
			List<Double> window = new ArrayList<Double>();

			// The first window
			for(int i = 0; i < windowSize; i++){
				window.addAll(encoding.get(di.getFeatureValueIndex(protein.get(i))));
			}
			examples.add(new ArrayList<Double>(window));

			// The rest of the windows
			for(int i = 1; i < protein.size() - windowSize + 1; i++){
				// Remove the previous window slot
				for(int j = 0; j < di.numFeatures; j++){
					window.remove(0);
				}
				// Add the next window slot
				window.addAll(encoding.get(di.getFeatureValueIndex(protein.get(i + windowSize - 1))));
				// Add window to example
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
		this.exampleSize = windowSize * (di.numFeatures);
	}

	// Generate list of values for one-hot encoding
	public List<List<Double>> generateEncoding(int encodingSize){
		List<List<Double>> list = new ArrayList<List<Double>>();

		List<Double> x = new ArrayList<Double>();
		for(int i = 1; i < encodingSize; i++){
			x.add(0.0);
		}
		x.add(1.0);

		list.add(new ArrayList<Double>(x));

		for(int i = 1; i < encodingSize; i++){
			x.remove(0);
			x.add(0.0);
			list.add(new ArrayList<Double>(x));			
		}

		return list;
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
			System.err.println("Size of input is: " + inputs.size());
			System.err.println("The size it should be is: " + numIn);
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
		return weights.clone();
	}
	
	public void setWeights(double[] weights){
		if(weights.length != numIn + 1){
			System.err.println("Wrong number of weights, setWeights fail");
			System.exit(-1);
		}

		this.weights = weights;
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
	// learningRate = learning rate of all the perceptrons, aka ETA
	public NeuralNetwork(int numInputs, int numHiddenUnits, int numClass, double learningRate){
		this.numHiddenUnits = numHiddenUnits;
		this.numClass = numClass;
		hiddenLayerOutputs = new ArrayList<Double>();
		hiddenLayer = new ArrayList<Perceptron>();

		for(int i = 0; i < numHiddenUnits; i++){
			hiddenLayer.add(new Perceptron(numInputs, "sig", learningRate));
			hiddenLayerOutputs.add(0.0);
		}

		outputLayer = new ArrayList<Perceptron>();
		outputs = new double[numClass];

		for(int i = 0; i < numClass; i++){
			outputLayer.add(new Perceptron(numHiddenUnits, "sig", learningRate));
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

	// Assumes that there's 2 layers only, the hidden layer and the output layer
	// This function returns a list that stores all the weights of the neural network
	public List<List<double[]>> exportWeights(){
		List<List<double[]>> layers = new ArrayList<List<double[]>>();
		List<double[]> hiddenWeights = new ArrayList<double[]>();
		List<double[]> outputWeights = new ArrayList<double[]>();

		for(Perceptron p : hiddenLayer){
			hiddenWeights.add(p.getWeights());
		}

		for(Perceptron p : outputLayer){
			outputWeights.add(p.getWeights());
		}

		layers.add(hiddenWeights);
		layers.add(outputWeights);

		return layers;
	}

	// Assumes that there's 2 layers only, the hidden layer and the output layer
	// Update the weights of this neural network with layers
	public void importWeights(List<List<double[]>> layers){
		List<double[]> hiddenWeights = layers.get(0);
		List<double[]> outputWeights = layers.get(1);

		if(numHiddenUnits != hiddenWeights.size()){
			System.err.println("Wrong number of hidden Perceptron when importing weight for the hidden layer");
			System.exit(-1);
		}

		if(numClass != outputWeights.size()){
			System.err.println("Wrong number of output Perceptron when importing weight for the output layer");
			System.exit(-1);
		}

		for(int i = 0; i < numHiddenUnits; i++){
			hiddenLayer.get(i).setWeights(hiddenWeights.get(i));
		}

		for(int i = 0; i < numClass; i++){
			outputLayer.get(i).setWeights(outputWeights.get(i));
		}
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

	public static void epochExperiment(int epoch, int epochUnit, ProteinData proteinData, int numHiddenUnits, double learningRate){
		NeuralNetwork nn = new NeuralNetwork(proteinData.trainList.exampleSize, numHiddenUnits, proteinData.di.numLabels, learningRate);

		double result = 0;
		double newResult = nn.test(proteinData.tuneList.examples);
		List<List<double[]>> optimalWeights = nn.exportWeights();

		for(int i = 0; i < epoch; i++){
			for(int j = 0; j < epochUnit; j++){
				nn.train(proteinData.trainList.examples);
			}

			System.out.println(i + ": " + nn.test(proteinData.testList.examples));
		}
	}

	public static void bestAccuracy(ProteinData proteinData){
		NeuralNetwork nn = new NeuralNetwork(proteinData.trainList.exampleSize, 3, proteinData.di.numLabels, 0.05);

		double result = 0;
		double newResult = nn.test(proteinData.tuneList.examples);
		List<List<double[]>> optimalWeights = nn.exportWeights();
		int patience = 50;
		int epoch = 1;

		// Early stopping
		// Loop stops after (i = patience) times if the result does not improve
		for(int i = 0; i < patience; i++){
			for(int j = 0; j < epoch; j++){
				nn.train(proteinData.trainList.examples);
			}

			newResult = nn.test(proteinData.tuneList.examples);
			if(newResult > result){
				// If new result is better, reset
				result = newResult;
				i = -1;

				// Keep track of the optimal weights
				optimalWeights = nn.exportWeights();
			}

			System.out.println(i + ": " + newResult);
		}

		// Final results
		nn.importWeights(optimalWeights);
		System.out.println("Tune: " + nn.test(proteinData.tuneList.examples));
		System.out.println("Final: " + nn.test(proteinData.testList.examples));
	}

	public static void main(String[] args){
		checkArgs(args);

		// Scanner that ignore comments for files
		CommentScanner inputScn = new CommentScanner(args[0]);

		ProteinData proteinData = new ProteinData(inputScn);

		// Epoch experiment, for plotting accuracy versus epoch graph
		// epochExperiment(1000, 2, proteinData, 3, 0.05);
		bestAccuracy(proteinData);
	}
}

// hidden:sig, output:sig, 100 hidden units, 0.001 learningRate, 10 patience, 5 epoch, Results: tune: 0.6299, final: 0.6107
// Best and fast, sig, sig, (2, 3, 4), (0.01, 0.005), 20, 5, Results: tune: 0.64, final: 0.62
// If ETA is very small, there's like an initial barrier that it has to break before the accuracy can increase