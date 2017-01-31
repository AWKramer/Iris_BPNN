package perceptron;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Perceptron {
	
	//define global variables of the perceptron 
	public double learningRate; 
	public double threshold; 
	public double maxIterations; 
	public int[] trainingOutputs; 
	public double[][] trainingInputs; 
	public double[][] testInputs; 
	public int[] testOutputs; 
	public double[] weights; 
	
	//constructor initialises given parameters 
	public Perceptron(double lr, double t, double maxIterations) {
		this.learningRate = lr; 
		this.threshold = t; 
		this.maxIterations = maxIterations; 
	}
	
	//reads in training inputs from train.txt file and stores them in trainingInputs array 
	//the 'type' parameter is used to specify which species we want to use to differentiate the data
	//the 'ignoreSetosa' flag is used in part four where we only want to separate Versicolor and Virginica data - so the Setosa data isn't used  
	public void populateTrainingInputs(String type, boolean ignoreSetosa) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader("train.txt")); 
		List<String> dataPoints = new ArrayList<String>(); 
		
		String s; 
		while ((s = br.readLine()) != null) { //read in data line by line 
			dataPoints.add(s); 
		}
		br.close(); 
		
		//initialise training data variables 
		this.trainingInputs = new double[120][5]; 
		this.trainingOutputs = new int[120]; 
		
		if (!ignoreSetosa) {
			for (int i = 0; i < 120; i++) {
				
				//split line to find individual feature values and output value 
				String[] points = dataPoints.get(i).split(","); 

				//add this as a data point to our array
				this.trainingInputs[i][0] = 1; //input is 1 for bias node 
				this.trainingInputs[i][1] = Double.parseDouble(points[0]); 
				this.trainingInputs[i][2] = Double.parseDouble(points[1]);
				this.trainingInputs[i][3] = Double.parseDouble(points[2]);
				this.trainingInputs[i][4] = Double.parseDouble(points[3]);
				
				//add the corresponding correct output for the data point
				this.trainingOutputs[i] = (points[4].contains(type)) ? 1 : 0; 
			}
		} else {
			for (int i = 0; i < 120; i++) {
				
				//split line to find individual feature values and output value 
				String[] points = dataPoints.get(i).split(","); 

				if (points[4].contains("setosa")) {
					//do nothing; we want to ignore setosa data points for part 4 
				} else {
					//add this as a data point to our array 
					this.trainingInputs[i][0] = 1;
					this.trainingInputs[i][1] = Double.parseDouble(points[0]); 
					this.trainingInputs[i][2] = Double.parseDouble(points[1]);
					this.trainingInputs[i][3] = Double.parseDouble(points[2]);
					this.trainingInputs[i][4] = Double.parseDouble(points[3]);
					
					//add the corresponding correct output for the data point
					this.trainingOutputs[i] = (points[4].contains(type)) ? 1 : 0; 
				} 
			}
		}
	}
	
	//reads in test data from test.txt, splits data line by line and stores it in the testInputs array 
	public void populateTestInputs(String type, boolean ignoreSetosa) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader("test.txt")); 
		List<String> dataPoints = new ArrayList<String>(); 
		
		String s; 
		while ((s = br.readLine()) != null) {
			dataPoints.add(s); 
		}
		br.close(); 
		
		this.testInputs = new double[30][5]; 
		this.testOutputs = new int[30]; 
		
		if (!ignoreSetosa) {
			for (int i = 0; i < 30; i++) {
				//split line to find individual feature values 
				String[] points = dataPoints.get(i).split(","); 
				
				//add this as a data point to our array 
				this.testInputs[i][0] = 1; //input is 1 for bias node 
				this.testInputs[i][1] = Double.parseDouble(points[0]); 
				this.testInputs[i][2] = Double.parseDouble(points[1]);
				this.testInputs[i][3] = Double.parseDouble(points[2]);
				this.testInputs[i][4] = Double.parseDouble(points[3]);
				
				//store corresponding correct output of this data point so we can calculate classification accuracy at the end 
				this.testOutputs[i] = (points[4].contains(type)) ? 1 : 0; 
			}
		} else {
			for (int i = 0; i < 30; i++) {
				//split line to find individual feature values 
				String[] points = dataPoints.get(i).split(","); 
				
				if (points[4].contains("setosa")) {
					//do nothing again 
				} else {
					//add this as a data point to our array 
					this.testInputs[i][0] = 1; //input is 1 for bias node 
					this.testInputs[i][1] = Double.parseDouble(points[0]); 
					this.testInputs[i][2] = Double.parseDouble(points[1]);
					this.testInputs[i][3] = Double.parseDouble(points[2]);
					this.testInputs[i][4] = Double.parseDouble(points[3]);
					
					//store correct output for data point 
					this.testOutputs[i] = (points[4].contains(type)) ? 1 : 0; 
				}
			}
		}
	}
	
	//initialises the weight array to random values for weights 1-4 and -threshold for weight 0 which is the bias node 
	public void initialiseWeights() {
		weights = new double[5]; 
		weights[0] = -this.threshold; //bias node 
		weights[1] = Math.random(); 
		weights[2] = Math.random(); 
		weights[3] = Math.random(); 
		weights[4] = Math.random(); 
	}
	
	public void trainErrorCorrection() {
		int iteration = 0; 
		double MSE = 0;  
		do { 
			iteration++; 
			MSE = 0; 
			for (int i = 0; i < 120; i++) {
				double[] dataPoint = this.trainingInputs[i];  
				int dataPointOutput = calculateOutput(dataPoint); //sum of weight * input 
				int dataPointError = this.trainingOutputs[i] - dataPointOutput; //desired output - predicted output 
				MSE += (Math.pow(dataPointError, 2)) / dataPoint.length; 
//				System.out.printf("\tMSE: " + MSE + "\n");				
				if (dataPointError != 0) { //if the error is not 0 then we need to update the weights 
					this.weights[0] += this.learningRate * dataPointError * dataPoint[0]; 
					this.weights[1] += this.learningRate * dataPointError * dataPoint[1];
					this.weights[2] += this.learningRate * dataPointError * dataPoint[2]; 
					this.weights[3] += this.learningRate * dataPointError * dataPoint[3];
					this.weights[4] += this.learningRate * dataPointError * dataPoint[4]; 
				} 
 
			}
			
		} while (MSE > 0 && iteration < this.maxIterations);  
		
		System.out.printf("Finished training after %d iterations.\n", iteration); 
		
	}
	
	public void trainSimpleFeedback() {
		int iteration = 0;
		int allCorrect = 0; 
		do { 
			iteration++; 
			for (int i = 0; i < 120; i++) {
				double[] dataPoint = this.trainingInputs[i]; 
				int dataPointOutput = calculateOutput(dataPoint);  
				int dataPointError = this.trainingOutputs[i] - dataPointOutput; 
				if (dataPointError == -1) { //if the error is not 0 then we need to update the weights. If the error is -1 then we use - 
					this.weights[0] -= this.learningRate * dataPoint[0]; 
					this.weights[1] -= this.learningRate * dataPoint[1];
					this.weights[2] -= this.learningRate * dataPoint[2]; 
					this.weights[3] -= this.learningRate * dataPoint[3]; 
					this.weights[4] -= this.learningRate * dataPoint[4]; 
					allCorrect = 0; 
				} else if (dataPointError == 1) {  //if the error is 1 then we use + 
					this.weights[0] += this.learningRate * dataPoint[0]; 
					this.weights[1] += this.learningRate * dataPoint[1];
					this.weights[2] += this.learningRate * dataPoint[2]; 
					this.weights[3] += this.learningRate * dataPoint[3]; 
					this.weights[4] += this.learningRate * dataPoint[4]; 
					allCorrect = 0;				
				} else {
					allCorrect++; 
				}
			}
			
		} while (allCorrect < this.trainingInputs.length && iteration <= this.maxIterations);  
		
		System.out.printf("Finished training after %d iterations.\n", iteration); 
		
	}
	
	//calculates the output of the Perceptron given the input 
	private int calculateOutput(double[] dataPoint) {
		double output = (dataPoint[0] * this.weights[0]) + (dataPoint[1] * this.weights[1]) + (dataPoint[2] * this.weights[2]) + (dataPoint[3] * this.weights[3]) + (dataPoint[4] * this.weights[4]); 
		return (output > 0) ? 1 : 0; 
	}
	
	//only called after training the neuron 
	//takes each testInput data point and runs it through calculateOutput to get the classification of the data point
	//stores the classification of the data point in an array for later analysis 
	public int[] test() {
		int[] classes = new int[30]; 
		for (int i = 0; i < 30; i++) {
			double[] testPoint = this.testInputs[i]; 
			classes[i] = calculateOutput(testPoint); 
		}
		return classes; 
	}	

} 
	
