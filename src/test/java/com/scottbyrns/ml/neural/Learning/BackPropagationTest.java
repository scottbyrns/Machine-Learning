package com.scottbyrns.ml.neural.Learning;

import com.scottbyrns.ml.datasets.DefaultPattern;
import com.scottbyrns.ml.datasets.DefaultPatternSet;
import com.scottbyrns.ml.datasets.PatternSet;
import com.scottbyrns.ml.neural.Activation.ActivationFunctionSigmoid;
import com.scottbyrns.ml.neural.DefaultFeedForwardNeuralNetwork;
import org.junit.Before;
import org.junit.Test;

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

        rprop.setTargetError(0.001);
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

    @Test
    public void test () {

    }


}
