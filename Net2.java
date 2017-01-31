package perceptron;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Net2 {
	
	//divides Iris-versicolour and Iris-virginica as best possible using Adaline error correction learning 
	public static void main(String[] args) throws IOException {
		Perceptron p = new Perceptron(1, 0.9, 5000); 
		
		System.out.println("***********\nPart four - training an Adaline with error correction learning\n***********");
		
		System.out.println("Populating training data..."); 
		p.populateTrainingInputs("virginica", true); //divide input data - virginica -> 1, versicolor -> 0. Ignore setosa data. 
		
		System.out.println("Populating test data...");
		p.populateTestInputs("virginica", true); //same parameters as populating training input 
		
		System.out.println("Initialising weights...");
		p.initialiseWeights(); 
		System.out.println("Initial weight values:"); 
		for (int i = 0; i < p.weights.length; i++) {
			System.out.printf("\tFeature %d weight: %f\n", i+1, p.weights[i]); 
		}
		
		System.out.println("Training...");
		p.trainErrorCorrection(); //use error correction algorithm  
		
		System.out.println("Final weight values:"); 
		for (int i = 0; i < p.weights.length; i++) {
			System.out.printf("\tFeature %d weight: %f\n", i+1, p.weights[i]); 
		}
		
		System.out.println("Predicting classes of test points (ignoring setosa)...");
		int[] classes = p.test(); 
		int correct = 0; 
		for (int i = 10; i < 30; i++) {
			System.out.printf("\tTest point: ");
			for (double d : p.testInputs[i]) {
				System.out.printf("%f ", d); 
			}
			if (classes[i] == p.testOutputs[i]) {
				System.out.printf("Class: %d - Correct\n", classes[i]);
				correct++; 
			} else {
				System.out.printf("Class: %d - Incorrect\n", classes[i]);
			}
			
		}
		System.out.printf("Total correct: %d\nTotal incorrect: %d\n", correct, 20 - correct);
		double acc = (correct / 20.0) * 100; 
		System.out.printf("Classification accuracy: %f%%", acc); 
		
		writeOutputToFile(p, classes); 
		
	}
	
	private static void writeOutputToFile(Perceptron p, int[] classes) throws IOException { 
		StringBuilder sb = new StringBuilder(); 
		for (int i = 10; i < p.testInputs.length; i++) { //starts at 10 to ignore the setosa data points which aren't classified 
			for (int j = 1; j < p.testInputs[i].length; j++) { //start at 1 to ignore the added bias node 
				sb.append(p.testInputs[i][j]).append(" "); 
			}
			sb.append("Predicted class: ").append(classes[i]).append("\n"); 
		}
	    Files.write(Paths.get("DataPointPredictionsPartFour.txt"), sb.toString().getBytes());
	}

}
