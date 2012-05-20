package com.scottbyrns.ml.neural.Learning;

import com.scottbyrns.ml.datasets.DefaultPattern;
import com.scottbyrns.ml.datasets.DefaultPatternSet;
import com.scottbyrns.ml.datasets.PatternSet;
import com.scottbyrns.ml.neural.Activation.ActivationFunctionSigmoid;
import com.scottbyrns.ml.neural.DefaultFeedForwardNeuralNetwork;
import org.junit.Before;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.util.Iterator;
import java.util.List;
import java.util.Vector;

/**
 * Created by scott
 * Date: Nov 13, 2010
 * Time: 1:44:47 PM
 */
public class BackPropagationTest {

    FeedForwardNetworkLearningAlgorithm rprop;
    PatternSet patternSet;

    @Before
    public void setup () {
        patternSet = new DefaultPatternSet();
        rprop = new BackPropagation(new DefaultFeedForwardNeuralNetwork(3, new int[]{3}, 1, new ActivationFunctionSigmoid()));
    }

    /**
     * Train an XOR pattern.
     */
    @Test
    public void trainXOR () {
        patternSet.addPattern(new DefaultPattern("0;0;0", "0"));
		patternSet.addPattern(new DefaultPattern("1;0;0", "1"));
		patternSet.addPattern(new DefaultPattern("1;1;0", "1"));
        patternSet.addPattern(new DefaultPattern("1;1;1", "0"));
        patternSet.addPattern(new DefaultPattern("0;1;1", "1"));
        patternSet.addPattern(new DefaultPattern("0;0;1", "1"));
        patternSet.addPattern(new DefaultPattern("0;1;0", "1"));
        patternSet.addPattern(new DefaultPattern("1;0;1", "1"));

        rprop.setTargetError(0.0001);
        rprop.setPatternSet(patternSet);
        rprop.setLearningStrategy(LearningStrategy.Memorize);

        rprop.startTraining();
        while (rprop.isRunning()) {
            try {
                Thread.sleep(100);
            }
            catch (Throwable e) {

            }
        }

        /**
         * @TODO make testable
         */
		// Test the network's accuracy and return the output to a string
		String output = new StringTester(rprop.getNetwork()).test(patternSet);


		// Print the string with the output
		System.out.println(output);
    }

    /**
     * Train an XNOR pattern.
     */
    @Test
    public void trainXNOR () {


       patternSet.addPattern(new DefaultPattern("0;0;0", "1"));
       patternSet.addPattern(new DefaultPattern("0;0;1", "0"));
       patternSet.addPattern(new DefaultPattern("0;1;0", "0"));
       patternSet.addPattern(new DefaultPattern("0;1;1", "1"));
       patternSet.addPattern(new DefaultPattern("1;0;0", "0"));
       patternSet.addPattern(new DefaultPattern("1;0;1", "1"));
       patternSet.addPattern(new DefaultPattern("1;1;0", "1"));
       patternSet.addPattern(new DefaultPattern("1;1;1", "0"));

        rprop.setTargetError(0.0001);
        rprop.setPatternSet(patternSet);
        rprop.setLearningStrategy(LearningStrategy.Memorize);

        rprop.startTraining();
        while (rprop.isRunning()) {
            try {
                Thread.sleep(100);
            }
            catch (Throwable e) {

            }
        }

        /**
         * @TODO make testable
         */
		// Test the network's accuracy and return the output to a string
		String output = new StringTester(rprop.getNetwork()).test(patternSet);


		// Print the string with the output
		System.out.println(output);
    }

//    @Test
    public void testImage () {
        try {
            File file;

            Vector<Double> patternData = new Vector<Double>();
            BufferedImage image, image2;
            Raster raster;

            image = ImageIO.read(new File("/Users/scott/nueral/1.png"));
            raster = image.getRaster();

            image2 = new BufferedImage(raster.getWidth(), raster.getHeight(), BufferedImage.TYPE_INT_RGB);
            int img = 6;
            while (--img > 0) {

                image = ImageIO.read(new File("/Users/scott/nueral/" + (img%2 == 0 ? "a" : "b") + ".png"));


                raster = image.getRaster();

                double[][][] pixelData = new double[raster.getWidth()][raster.getHeight()][3];
                int[][] intData = new int[raster.getWidth()][raster.getHeight()];

                patternData = new Vector<Double>();

                for (int i = 0; i < raster.getWidth(); i += 1) {
                    for (int j = 0; j < raster.getHeight(); j += 1) {
                        double[] a = raster.getPixel(i, j, new double[4]);
                        pixelData[i][j] = a;

                        intData[i][j] = (int)(256*256*a[0]+256*a[2]+a[1]);

                        patternData.add((((1./65792D) * ((((((int)a[0]) << 8) + (int)a[1]) << 8) + a[2]))));
//
//                        if (img == 25) {
//                            System.out.println(((1./65792D) * ((((((int)a[0]) << 8) + (int)a[1]) << 8) + a[2])) * 65792);
//                            image2.setRGB(i, j, (int)(((1./65792D) * ((((((int)a[0]) << 8) + (int)a[1]) << 8) + a[2]))) * 65792);
//                        }


                    }
                }

                patternSet.addPattern(new DefaultPattern((Vector<Double>)patternData.clone(), (Vector<Double>)patternData.clone()));
            }


            rprop = new BackPropagation(new DefaultFeedForwardNeuralNetwork(patternData.size(), new int[]{(int)(patternData.size())}, patternData.size(), new ActivationFunctionSigmoid()), 5);
            rprop.setLearningStrategy(LearningStrategy.Memorize);

            rprop.setTargetError(0.001);
            rprop.setPatternSet(patternSet);

            rprop.startTraining();
            while (rprop.isRunning()) {
                try {
                    Thread.sleep(1000);
//                    System.out.println(rprop.getCurrentEpoch());
                }
                catch (Throwable e) {

                }
            }


            file = new File("/tmp/im.png");
            ImageIO.write(image2, "png", file);

            image = new BufferedImage(raster.getWidth(), raster.getHeight(), BufferedImage.TYPE_INT_RGB);

            List<Double> prediction = rprop.getNetwork().getPrediction(patternSet.getTrainingSet().get(0).getOutput());
            prediction = rprop.getNetwork().getPrediction(prediction);
            Iterator<Double> predictionIterator = prediction.iterator();
            for (int i = 0; i < raster.getWidth(); i += 1) {
                for (int j = 0; j < raster.getHeight(); j += 1) {
//
//                    int red = (int)(256*predictionIterator.next());
//                    int green = (int)(256*predictionIterator.next());
//                    int blue = (int)(256*predictionIterator.next());

                    int color = (int)(predictionIterator.next() * ((((256) << 8) + 256) << 8) + 256);
//                    System.out.println(image.getRGB(i, j) + " " + color);
                    image.setRGB(i, j, color);

//                    System.out.println(image.getRGB(i, j) + "=" + color);

                }
            }

            file = new File("/tmp/1.jpg");
            ImageIO.write(image, "jpg", file);
            prediction = rprop.getNetwork().getPrediction(prediction);
            predictionIterator = prediction.iterator();
            for (int i = 0; i < raster.getWidth(); i += 1) {
                for (int j = 0; j < raster.getHeight(); j += 1) {
//
//                    int red = (int)(256*predictionIterator.next());
//                    int green = (int)(256*predictionIterator.next());
//                    int blue = (int)(256*predictionIterator.next());

                    int color = (int)(predictionIterator.next() * 65792);
//                    System.out.println(image.getRGB(i, j) + " " + color);
                    image.setRGB(i, j, color);

//                    System.out.println(image.getRGB(i, j) + "=" + color);

                }
            }

            double[] predictionData = new double[prediction.size()];

            for (int i = 0; i < prediction.size(); i += 1) {
                predictionData[i] = prediction.get(i);
            }

            patternSet.addPattern(new DefaultPattern(predictionData, predictionData));

            rprop = new BackPropagation(rprop.getNetwork(), 5);
            rprop.setLearningStrategy(LearningStrategy.Memorize);

            rprop.setTargetError(0.001);


            rprop.setPatternSet(patternSet);

            rprop.startTraining();
            while (rprop.isRunning()) {
                try {
                    Thread.sleep(1000);
//                    System.out.println(rprop.getCurrentEpoch());
                }
                catch (Throwable e) {

                }
            }

            file = new File("/tmp/2.jpg");
            ImageIO.write(image, "jpg", file);
            prediction = rprop.getNetwork().getPrediction(prediction);
            predictionIterator = prediction.iterator();
            for (int i = 0; i < raster.getWidth(); i += 1) {
                for (int j = 0; j < raster.getHeight(); j += 1) {
//
//                    int red = (int)(256*predictionIterator.next());
//                    int green = (int)(256*predictionIterator.next());
//                    int blue = (int)(256*predictionIterator.next());

                    int color = (int)(predictionIterator.next() * 65792);
//                    System.out.println(image.getRGB(i, j) + " " + color);
                    image.setRGB(i, j, color);

//                    System.out.println(image.getRGB(i, j) + "=" + color);

                }
            }

            predictionData = new double[prediction.size()];

            for (int i = 0; i < prediction.size(); i += 1) {
                predictionData[i] = prediction.get(i);
            }

            patternSet.addPattern(new DefaultPattern(predictionData, predictionData));
            rprop = new BackPropagation(rprop.getNetwork(), 5);
            rprop.setLearningStrategy(LearningStrategy.Memorize);

            rprop.setTargetError(0.001);
            rprop.setPatternSet(patternSet);

            rprop.startTraining();
            while (rprop.isRunning()) {
                try {
                    Thread.sleep(1000);
//                    System.out.println(rprop.getCurrentEpoch());
                }
                catch (Throwable e) {

                }
            }

            file = new File("/tmp/3.jpg");
            ImageIO.write(image, "jpg", file);
            prediction = rprop.getNetwork().getPrediction(prediction);

            predictionIterator = prediction.iterator();
            for (int i = 0; i < raster.getWidth(); i += 1) {
                for (int j = 0; j < raster.getHeight(); j += 1) {
//
//                    int red = (int)(256*predictionIterator.next());
//                    int green = (int)(256*predictionIterator.next());
//                    int blue = (int)(256*predictionIterator.next());

                    int color = (int)(predictionIterator.next() * 65792);
//                    System.out.println(image.getRGB(i, j) + " " + color);
                    image.setRGB(i, j, color);

//                    System.out.println(image.getRGB(i, j) + "=" + color);

                }
            }

            predictionData = new double[prediction.size()];

            for (int i = 0; i < prediction.size(); i += 1) {
                predictionData[i] = prediction.get(i);
            }

            patternSet.addPattern(new DefaultPattern(predictionData, predictionData));
            rprop = new BackPropagation(rprop.getNetwork(), 5);
            rprop.setLearningStrategy(LearningStrategy.Memorize);

            rprop.setTargetError(0.001);
            rprop.setPatternSet(patternSet);

            rprop.startTraining();
            while (rprop.isRunning()) {
                try {
                    Thread.sleep(1000);
//                    System.out.println(rprop.getCurrentEpoch());
                }
                catch (Throwable e) {

                }
            }

            file = new File("/tmp/4.jpg");
            ImageIO.write(image, "jpg", file);
            prediction = rprop.getNetwork().getPrediction(prediction);
            predictionIterator = prediction.iterator();
            for (int i = 0; i < raster.getWidth(); i += 1) {
                for (int j = 0; j < raster.getHeight(); j += 1) {
//
//                    int red = (int)(256*predictionIterator.next());
//                    int green = (int)(256*predictionIterator.next());
//                    int blue = (int)(256*predictionIterator.next());

                    int color = (int)(predictionIterator.next() * 65792);
//                    System.out.println(image.getRGB(i, j) + " " + color);
                    image.setRGB(i, j, color);

//                    System.out.println(image.getRGB(i, j) + "=" + color);

                }
            }

            predictionData = new double[prediction.size()];

            for (int i = 0; i < prediction.size(); i += 1) {
                predictionData[i] = prediction.get(i);
            }

            patternSet.addPattern(new DefaultPattern(predictionData, predictionData));
            rprop = new BackPropagation(rprop.getNetwork(), 5);
            rprop.setLearningStrategy(LearningStrategy.Memorize);

            rprop.setTargetError(0.001);
            rprop.setPatternSet(patternSet);

            rprop.startTraining();
            while (rprop.isRunning()) {
                try {
                    Thread.sleep(1000);
//                    System.out.println(rprop.getCurrentEpoch());
                }
                catch (Throwable e) {

                }
            }

            file = new File("/tmp/5.jpg");
            ImageIO.write(image, "jpg", file);
            predictionIterator = prediction.iterator();
            for (int i = 0; i < raster.getWidth(); i += 1) {
                for (int j = 0; j < raster.getHeight(); j += 1) {
//
//                    int red = (int)(256*predictionIterator.next());
//                    int green = (int)(256*predictionIterator.next());
//                    int blue = (int)(256*predictionIterator.next());

                    int color = (int)(predictionIterator.next() * 65792);
//                    System.out.println(image.getRGB(i, j) + " " + color);
                    image.setRGB(i, j, color);

//                    System.out.println(image.getRGB(i, j) + "=" + color);

                }
            }

            file = new File("/tmp/6.jpg");
            ImageIO.write(image, "jpg", file);

        }
        catch (Exception e) {
            e.printStackTrace();
        }


    }

    @Test
    public void test () {

    }


}
