package baguette;

import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.event.WindowAdapter;
import java.awt.GridLayout;
import java.awt.BorderLayout;
import javax.swing.JLabel;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.awt.Color;
import java.awt.Component;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import javax.swing.JFrame;
import javax.swing.JPanel;

public class Helpers {
    /**
     * @brief Reads all the bytes of a file
     *
     * @param filename the path of the file to read
     * 
     * @return the bytes contained in the file in an array
     */
    public static byte[] readBinaryFile(String filename) {
        Path path = Paths.get(filename);
        byte[] data = null;
        try {
            data = Files.readAllBytes(path);
        } catch (IOException io) {
            System.out.println("Could not open file");
            System.out.println(io.getMessage());
        }
        return data;
    }

    /**
     * @brief Writes some bytes to a file
     *
     * @param filename the path of the file to write to
     * @param data     the bytes to write
     */
    public static void writeBinaryFile(String filename, byte[] data) {
        Path path = Paths.get(filename);
        try {
            Files.write(path, data);

        } catch (IOException io) {
            System.out.println("Could not write to file");
            System.out.println(io.getMessage());
        }
    }

    /**
     * @brief Writes a byte as the sequence of its bits using big endian convention.
     *
     * @param b the byte to write
     * 
     * @return the sequence of bits forming b
     */
    public static String byteToBinaryString(byte b) {
        return Integer.toBinaryString((0xFF & b) + 0x100).substring(1);
    }

    /**
     * @brief Creates a signed byte from a string of bits
     * 
     * @param bits the string of bytes to interpret
     * 
     * @return the signed byte corresponding to bits
     */
    public static byte binaryStringToByte(String bits) {
        if (bits.length() > 8)
            throw new IllegalArgumentException("Invalid sequence of bits");

        int res = Integer.parseInt(bits, 2);
        return (byte) res;
    }

    /**
     * @brief Interprets the provided binary string as a signed byte
     * 
     * @param bits the string of bytes to interpret
     * 
     * @return the number obtained interpreting bits as a signed byte
     */
    public static String interpretSigned(String bits) {
        return "" + binaryStringToByte(bits);
    }

    /**
     * @brief Interprets the provided binary string as an unsigned byte
     * 
     * @param bits the string of bytes to interpret
     * 
     * @return the number obtained interpreting bits as an unsigned byte
     */
    public static String interpretUnsigned(String bits) {
        if (bits.length() > 8)
            throw new IllegalArgumentException("Invalid sequence of bits");

        return "" + Integer.parseUnsignedInt(bits, 2);
    }

    /**
     * @brief Displays a grid with the provided images and their labels. Only rows *
     *        columns images will be displayed (even if more are provided) Correct
     *        labels are highlighted in green, wrong ones in red.
     *
     * @param title      the title of the window
     * @param tensor     the tensor containing the images to draw
     * @param predictions     the labels to show under each image
     * @param trueLabels the true labels of the images. If label == trueLabel the
     *                   text is in a green box, otherwise it is in a red one
     * @param rows       the number of rows of images in the grid
     * @param columns    the number of rows of images in the grid
     */
    public static void show(String title, double[][] images, int rows, int columns) {
        JFrame frame = initFrame(rows, columns, title);
        for (int i = 0; i < Math.min(rows * columns, images.length); i++) {
            frame.add(imagePanel(toBufferedImage(images[i])));
        }
        drawFrame(frame);
    }

    private static JFrame initFrame(int rows, int columns, String title) {
        final JFrame frame = new JFrame(title);
        frame.setLayout(new GridLayout(rows, columns));
        return frame;
    }

    private static void drawFrame(JFrame frame) {
        frame.pack();
        // Register closing event
        frame.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
        frame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                frame.setVisible(false);
                synchronized (frame) {
                    frame.notifyAll();
                }
            }
        });
        // Show this frame
        frame.setVisible(true);

        // Wait for close operation
        try {
            synchronized (frame) {
                while (frame.isVisible())
                    frame.wait();
            }
        } catch (InterruptedException e) {
            // Empty on purpose
        }
        frame.dispose();
    }

    private static BufferedImage toBufferedImage(double[] testImage) {
        int width = 28;
        int height = 28;
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                image.setRGB(x, y, new Color((int) (255 *  testImage[y * width + x]), (int) (255 *  testImage[y * width + x]), (int) (255 *  testImage[y * width + x])).getRGB());
            }
        }
        return image;
    }

    private static JPanel imagePanel(BufferedImage im) {
        @SuppressWarnings("serial")
        JPanel imPanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                g.drawImage(im, 0, 0, getWidth(), getHeight(), null, null);
            }
        };
        JPanel p = new JPanel();
        p.setLayout(new BorderLayout());
        p.add(imPanel, BorderLayout.CENTER);
        p.setPreferredSize(new Dimension(Math.max(im.getWidth(), 56), Math.max(im.getHeight(), 56)));
        return p;
    }

    private static JPanel imagePanel(BufferedImage im, double label) {
        JPanel p = imagePanel(im);
        JLabel labelContainer = new JLabel("" + label, JLabel.CENTER);
        p.add(labelContainer, BorderLayout.SOUTH);
        return p;
    }

    private static JPanel imagePanel(BufferedImage im, double label, double trueLabel) {
        JPanel res = imagePanel(im, label);
        JLabel l = (JLabel) res.getComponent(1);
        l.setBackground(label == trueLabel ? Color.GREEN : Color.RED);
        l.setOpaque(true);
        return res;
    }
}
