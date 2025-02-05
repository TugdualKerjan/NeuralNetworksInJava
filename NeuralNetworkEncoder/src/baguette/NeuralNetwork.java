package baguette;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.util.Random;
import java.util.Scanner;

import javax.swing.JFrame;
import javax.swing.WindowConstants;

public class NeuralNetwork {
	//LOAD FILES
	static double[][] testImages = parseIDXimages(Helpers.readBinaryFile("datasets/1000-per-digit_images_train")) ;
	static double[][] trainImages = parseIDXimages(Helpers.readBinaryFile("datasets/10k_images_test")) ;

	//[LAYERS][NEURONS IN THAT LAYER][WEIGHTS CONNECTING THAT NEURON TO PREVIOUS LAYER]
	static double[][][] network;
	static double[][] output;
	static double[][] bias;
	static double[][] neuronError;
	static double[][] outputDerivative;

	static double learningRate;
	static int LAYER_SIZE;

	//Random thing that doesn't work
	static Random random; 

	//Panel on which to draw
	static neuronPanel panel;

	//Speed of the iterations in ms
	static int waitTime = 0;

	public static void main(String [] args) {
		//Create random object
		random = new Random();
		//Create neural network
		initializeNetwork();
	}

	public static void initializeNetwork() {
		//Auto put learningRate to 1.0
		learningRate = 0.2;

		//Setup the neural network
		Scanner scan = new Scanner(System.in);
		System.out.println("Amount of hidden layers wanted in the neural network");
		int hiddenLayers = scan.nextInt();
		if(hiddenLayers < 1) {
			System.out.println("Amount of hidden layers too low, system set hidden layers to 1");
			hiddenLayers = 1;
		}

		LAYER_SIZE = hiddenLayers + 2;

		//Includes the output and input layer
		network = new double[LAYER_SIZE][][];
		output = new double[LAYER_SIZE][];
		neuronError = new double[LAYER_SIZE][];
		outputDerivative = new double[LAYER_SIZE][];
		bias = new double[LAYER_SIZE][];

		//Special since there needs to be 28 * 28 weights for the first layer (input layer is before it) and in neuronValues
		network[0] = new double[28 * 28][1];
		output[0] = new double[28 * 28];
		neuronError[0] = new double[28 * 28];
		outputDerivative[0] = new double[28 * 28];
		bias[0] = new double[28 * 28];

		//Asks for amount of neurons for every hidden layer, and in neuronValues
		for(int i = 1; i <= hiddenLayers; i++) {
			System.out.println("Amount of neurons wanted in hidden layer " + i);
			int neurons = scan.nextInt();
			network[i] = new double[neurons][network[i - 1].length];
			output[i] = new double[neurons];
			neuronError[i] = new double[neurons];
			outputDerivative[i] = new double[neurons];
			bias[i] = new double[neurons];
		}

		//Sets amount of output neurons to ten in network and neuronValues
		network[LAYER_SIZE-1] = new double[trainImages[0].length][network[hiddenLayers].length];
		output[LAYER_SIZE-1] = new double[trainImages[0].length];
		neuronError[LAYER_SIZE-1] = new double[trainImages[0].length];
		outputDerivative[LAYER_SIZE-1] = new double[trainImages[0].length];
		bias[LAYER_SIZE-1] = new double[trainImages[0].length];

		//Closes scanner
		scan.close();

		//Randomise the weights
		randomizeWeightsAndBiases();

		//Display the network
		show(network);

		//Train the network on some data
		train(100, 1000);

		double[][] predictions = new double[700][28*28];

		for (int i = 0 ; i < 700 ; i++) {
			forwardPropagation(testImages[i]);
			predictions[i] = output[network.length - 1];
		}

		accuracy(predictions, testImages);

		Helpers.show("Neural network", testImages, 20, 35);
	}

	/**
	 * Train the network with batches (Don't know how to do yet)
	 *
	 * @param image vector, inputed into the neural network
	 */
	private static void train(int batchSize, int amountOfBatches) {
		double[][][] batches = new double[amountOfBatches][batchSize][];
		for(int batch = 0; batch < amountOfBatches; batch++) {
			for(int image = 0; image < batchSize; image++) {
				int randomIndex = new Random().nextInt(trainImages.length);
				batches[batch][image] = trainImages[randomIndex];
			}
		}

		for(int batch = 0; batch < amountOfBatches; batch++) {
			//Sum the errors of the batch to then apply gradient descent
			double sumOfErrors = 0;
			for(int image = 0; image < batchSize; image++) {
				//Propagate forward and return the MSE
				sumOfErrors += forwardPropagation(batches[batch][image]);
				backPropagation(batches[batch][image]);
				updateWeights();
			}
			//Modify the weights after the batch
			System.out.println(batch + ":MSE of the batch: " + sumOfErrors / batchSize);
		}
	}

	/**
	 * Takes an image as input and returns the output (10 big vector)
	 *
	 * @param image vector, inputed into the neural network
	 */
	private static void updateWeights() {
		for(int layer = 1; layer < LAYER_SIZE; layer++) {
			for(int neuron = 0; neuron < network[layer].length; neuron++) {

				double delta = - learningRate * neuronError[layer][neuron];

				for(int weight = 0; weight < network[layer][neuron].length; weight++) {

					network[layer][neuron][weight] += delta * output[layer - 1][weight];

				}

			}
		}
	}

	/**
	 * Takes an image as input and returns the output (10 big vector)
	 *
	 * @param image vector, inputed into the neural network
	 */
	private static void backPropagation(double[] target) {	
		//Calculate the output layer first, which is calculated differently
		for(int neuron = 0; neuron < network[LAYER_SIZE - 1].length; neuron++) {
			neuronError[LAYER_SIZE - 1][neuron] = (output[LAYER_SIZE - 1][neuron] - target[neuron]) * outputDerivative[LAYER_SIZE - 1][neuron];
		}

		//Proceed from forward to backwards, already having affected the last layer, thus layer - 2
		for(int layer = LAYER_SIZE - 2; layer >= 0; layer--) {
			//This works as it should: Sum the weights of a neuron to the nextNeurons times the respective error of the next neuron, and time it by outDeriv to get neuronError
			for(int neuron = 0; neuron < network[layer].length; neuron++) {
				double sum = 0;
				for(int nextNeuron = 0; nextNeuron < network[layer + 1].length; nextNeuron++) {
					sum += network[layer + 1][nextNeuron][neuron] * neuronError[layer + 1][nextNeuron];
				}
				neuronError[layer][neuron] = sum * outputDerivative[layer][neuron];
			}
		}
	}

	/**
	 * Takes an image as input and returns the output (10 big vector)
	 *
	 * @param image vector, inputed into the neural network
	 */
	private static double forwardPropagation(double[] image) {
		assert image.length == 28*28;
		output[0] = image;

		//Set the values of the first layer to the values of the image vector
		for(int neuron = 0; neuron < network[0].length; neuron++) {
			outputDerivative[0][neuron] = output[0][neuron] * (1 - output[0][neuron]);
		}

		//Calculate the values of the neurons for that image until the last layer
		for(int layer = 1; layer < LAYER_SIZE; layer++) {
			for(int neuron = 0; neuron < network[layer].length; neuron++) {

				double totalOfThatNeuron = bias[layer][neuron];
				for(int weight = 0; weight < network[layer][neuron].length; weight++) {
					//Sum up all the outputs of the previous layer with their respective weights
					totalOfThatNeuron += output[layer - 1][weight] * network[layer][neuron][weight];
				}
				output[layer][neuron] = sigmoid(totalOfThatNeuron);
				outputDerivative[layer][neuron] = output[layer][neuron] * (1 - output[layer][neuron]);
			}
		}

		neuronPanel.neuronValues = output;
		neuronPanel.network = network;
		neuronPanel.input = image;

		//Distance between input and output image
		if(euclideanDistance(image, output[output.length-1]) < 0.17) {
			neuronPanel.correct = true;
		} else {
			neuronPanel.correct = false;
		}

		//Repaint the image
		panel.repaint();

		//Sleep time
		try {
			Thread.sleep(waitTime);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		//Return the values of the last layer of neurons, the output layer
		return MeanSquaredError(output[LAYER_SIZE - 1], image);
	}

	private static double MeanSquaredError(double[] output, double[] target) {
		double total = 0;
		for(int guess = 0; guess < output.length; guess++) {
			total += Math.pow(output[guess] - target[guess], 2);
		}
		total /= output.length;
		return total;
	}

	/**
	 * Applies the sigmoid function to the input
	 *
	 * @param sum, 
	 */
	private static double sigmoid(double sum) {
		return 1d / (1 + Math.exp(-sum));
	}

	/**
	 * Gives random values between -1.0 and 1.0 to the weights
	 *
	 * @param tensor, the tensor to which we randomise the weights
	 */
	private static void randomizeWeightsAndBiases() {
		for(int layer = 1; layer < LAYER_SIZE; layer++) {
			for(int neuron = 0; neuron < network[layer].length; neuron++) {
				for(int weight = 0; weight < network[layer][neuron].length; weight++) {
					network[layer][neuron][weight] = new Random().nextDouble() * 2 - 1d;
					bias[layer][neuron] = new Random().nextDouble() * 2 - 1d;
				}
			}
		}
	}

	/**
	 * Returns the array of images as double[][]
	 * 
	 * @param data the binary content of the file 
	 * 
	 * @return the array of images as doubles[width][height]
	 */
	public static double[][] parseIDXimages(byte[] data) {
		int amountOfImages = KNN.extractInt(data[4], data[5], data[6], data[7]);
		int heightOfImages = KNN.extractInt(data[8], data[9], data[10], data[11]);
		int widthOfImages = KNN.extractInt(data[12], data[13], data[14], data[15]);

		System.out.println("Amount: " + amountOfImages + ", height: " + heightOfImages + ", width: " + widthOfImages);

		double[][] tensor = new double[amountOfImages][widthOfImages * heightOfImages];
		for(int image = 0; image < amountOfImages; image++) {
			for(int y = 0; y < heightOfImages; y++) {
				for(int x = 0; x < widthOfImages; x++) {
					//For one image, there are 28 x 28 pixels, plus for one row, there are 28 pixels
					int displacement = image * (widthOfImages * heightOfImages) + y * heightOfImages + x;
					byte smallToBigEndian = (byte) 128;
					//data is between 0 and 256 with 0 being black, thus add 128 to make into -128 to 128 and then divide by 255 to get inbetween 0.0 and 1.0
					tensor[image][y * widthOfImages + x] = ((double) ((byte) (data[16 + displacement] + smallToBigEndian)) + 128.0) / 255.0;
				}
			}
		}
		return tensor;
	}

	/**
	 * Creates the window for the viewing of the neural network
	 *
	 * @param network, The network to display
	 */
	public static void show(double[][][] network) {
		JFrame frame = new JFrame("Neural network");
		frame.addKeyListener(new KeyListener() {
			@Override
			public void keyPressed(KeyEvent e) {
				if(e.getKeyCode() == KeyEvent.VK_UP) {
					NeuralNetwork.waitTime += 1000;
				} else if (e.getKeyCode() == KeyEvent.VK_DOWN) {
					if(NeuralNetwork.waitTime >= 1000) NeuralNetwork.waitTime -= 1000;
				}
			}

			@Override
			public void keyReleased(KeyEvent e) {

			}

			@Override
			public void keyTyped(KeyEvent e) {

			}

		});
		frame.setSize(1920, 1080);
		frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		frame.setResizable(false);
		panel = new neuronPanel(network);
		frame.add(panel);
		frame.pack();
		frame.setVisible(true);
		frame.setAlwaysOnTop(true);
	}

	/**
	 * Computes accuracy between two arrays of predictions
	 * 
	 * @param predictedImages the array of labels predicted by the algorithm
	 * @param trueImages      the array of true labels
	 * 
	 * @return the accuracy of the predictions. Its value is in [0, 1]
	 */
	public static double accuracy(double[][] predictedImages, double[][] trueImages) {
		double accuracy = 0;
		for(int prediction = 0; prediction < predictedImages.length; prediction++) {
			accuracy += euclideanDistance(predictedImages[prediction], trueImages[prediction]);
		} 
		accuracy /= predictedImages.length;
		System.out.println("Neural Network acc: " + accuracy * 100);
		return accuracy * 100;
	}


	/**
	 * @brief Computes the squared L2 distance of two images
	 * 
	 * @param a, b two images of same dimensions
	 * 
	 * @return the squared euclidean distance between the two images
	 */
	public static double euclideanDistance(double[] a, double[] b) {
		assert a.length == b.length;

		float distance = 0f;

		for(int x = 0; x < a.length; x++) {
			distance += (float) Math.pow(a[x] - b[x], 2);
		}

		return Math.sqrt(distance);
	}

}
