package perceptron;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Net1 {
	
	//classifies test input as either Iris-setosa or not using simple feedback training 
	public static void main(String[] args) throws IOException {
		Perceptron p = new Perceptron(1, 1, 10000); 
		
		System.out.println("***********\nPart two - training a Perceptron with simple feedback learning\n***********");
		
		System.out.println("Populating training data..."); 
		p.populateTrainingInputs("setosa", false); 
		
		System.out.println("Populating test data...");
		p.populateTestInputs("setosa", false);
		
		System.out.println("Initialising weights...");
		p.initialiseWeights(); 
		System.out.println("Initial weight values:"); 
		for (int i = 0; i < p.weights.length; i++) {
			System.out.printf("\tFeature %d weight: %f\n", i+1, p.weights[i]); 
		}
		
		System.out.println("Training...");
		p.trainSimpleFeedback(); //use simple feedback training algorithm 
		
		System.out.println("Final weight values:"); 
		for (int i = 0; i < p.weights.length; i++) {
			System.out.printf("\tFeature %d weight: %f\n", i+1, p.weights[i]); 
		}
		
		System.out.println("Predicting classes of test points...");
		int[] classes = p.test(); 
		int correct = 0; 
		for (int i = 0; i < 30; i++) {
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
		System.out.printf("Total correct: %d\nTotal incorrect: %d\nClassification accuracy: %f%%", correct, 30 - correct, (correct / 30.0) * 100); 
		
		writeOutputToFile(p, classes); 
		
	}
	
	private static void writeOutputToFile(Perceptron p, int[] classes) throws IOException { 
		StringBuilder sb = new StringBuilder(); 
		for (int i = 0; i < p.testInputs.length; i++) {
			for (int j = 1; j < p.testInputs[i].length; j++) { //start at 1 to ignore the added bias node 
				sb.append(p.testInputs[i][j]).append(" "); 
			}
			sb.append("Predicted class: ").append(classes[i]).append("\n"); 
		}
	    Files.write(Paths.get("DataPointPredictionsPartTwo.txt"), sb.toString().getBytes());
	}

}
