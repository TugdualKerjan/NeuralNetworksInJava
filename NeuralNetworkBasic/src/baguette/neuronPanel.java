package baguette;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;

import javax.swing.JPanel;

public class neuronPanel extends JPanel {

	private static final long serialVersionUID = 1L;

	public static double[][][] network;
	public static double[][] neuronValues;
	public static double[] input;

	public static boolean correct;

	public neuronPanel(double[][][] network) {
		neuronPanel.network = network;
		this.setPreferredSize(new Dimension(1920, 1080));
	}

	@Override
	protected void paintComponent(Graphics g) {
		super.paintComponent(g);

		if(network == null || neuronValues == null) return;
		int neuronRadiusSize = 10;
		int pixelImageSize = 5;

		int width = this.getWidth();
		int height = this.getHeight();
		
		if(correct) {
			g.setColor(new Color(230, 255, 230));
			g.fillRect(0, 0, width, height);
		} else {
			g.setColor(new Color(255, 230, 230));
			g.fillRect(0, 0, width, height);
		}
		g.setColor(Color.BLACK);
		
		//Account for the image
		double layerWidth = (double) width / (double) (network.length + 2);
		
		int xCenter = (int) layerWidth - (28/2) * pixelImageSize;
		int yCenter = (int) height/2 - (28/2) * pixelImageSize;
		
		//Draw image
		for(int y = 0; y < 28; y++) {
			for(int x = 0; x < 28; x++) {
				g.setColor(new Color((int) (255 * input[y * 28 + x]), (int) (255 * input[y * 28 + x]), (int) (255 * input[y * 28 + x])));
				g.fillRect(xCenter + x * pixelImageSize, yCenter + y * pixelImageSize, pixelImageSize, pixelImageSize);
			}
		}

		//Draw Weights
		for(int layer = 0; layer < network.length; layer++) {
			
			double neuronsHeight =  (double) height / (double) (network[layer].length + 1);
			
			for(int neuron = 0; neuron < network[layer].length; neuron++) {
				
				if(layer == 0) {
					
					double biggestWeight = 0;
					for(int weight = 0; weight < network[layer][neuron].length; weight++) {
						if(biggestWeight < Math.abs(network[layer][neuron][weight])) biggestWeight = Math.abs(network[layer][neuron][weight]);
						
					}
					
					for(int weight = 0; weight < network[layer][neuron].length; weight++) {
						g.setColor(new Color((int) (128 + 120 * (network[layer][neuron][weight] / biggestWeight)), 0, (int) (128 - 120 * (network[layer][neuron][weight]) / biggestWeight)));
						g.drawLine((int) layerWidth + (28/2) * pixelImageSize, height/2, (int) (layerWidth * (layer + 2)), (int) (neuronsHeight * (neuron + 1)));
					}
					continue;
				}
				
				double neuronsHeightBefore = (double) height / (double) (network[layer - 1].length + 1);

				double biggestWeight = 0;
				for(int weight = 0; weight < network[layer][neuron].length; weight++) {
					if(biggestWeight < Math.abs(network[layer][neuron][weight])) biggestWeight = Math.abs(network[layer][neuron][weight]);
					
				}
				
				for(int weight = 0; weight < network[layer][neuron].length; weight++) {
					g.setColor(new Color((int) (128 + 120 * (network[layer][neuron][weight] / biggestWeight)), 0, (int) (128 - 120 * (network[layer][neuron][weight]) / biggestWeight)));
					g.drawLine((int) (layerWidth * (layer + 1)), (int) (neuronsHeightBefore * (weight + 1)), (int) (layerWidth * (layer + 2)), (int) (neuronsHeight * (neuron + 1)));
				}
			}
		}

		//Draw neurons
		for(int layer = 0; layer < network.length; layer++) {
			double neuronsHeight =  (double) height / (double) (network[layer].length + 1);
			for(int neuron = 0; neuron < network[layer].length; neuron++) {
				//Sets the color of the neuron to white if the output value is high
				g.setColor(new Color((int) (255 *  neuronValues[layer][neuron]), (int) (255 * neuronValues[layer][neuron]), (int) (255 *  neuronValues[layer][neuron])));
				g.fillOval((int) ((layerWidth * (layer + 2)) - neuronRadiusSize), (int) ((neuronsHeight * (neuron + 1) - neuronRadiusSize)), neuronRadiusSize * 2, neuronRadiusSize * 2);
			}
		}
	}
}
