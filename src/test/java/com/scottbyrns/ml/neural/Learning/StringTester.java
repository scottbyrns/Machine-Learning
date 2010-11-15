package com.scottbyrns.ml.neural.Learning;

import com.scottbyrns.ml.datasets.Pattern;
import com.scottbyrns.ml.datasets.PatternSet;
import com.scottbyrns.ml.datasets.PatternType;
import com.scottbyrns.ml.neural.FeedForwardNeuralNetwork;

import java.util.List;

/**
 * Created by scott
 * Date: Nov 12, 2010
 * Time: 7:49:28 PM
 */
public class StringTester
{
	FeedForwardNeuralNetwork	network	= null;

	public StringTester(FeedForwardNeuralNetwork net)
	{
		this.network = net;
	}

	public String test(PatternSet pattern_set)
	{
		String output_text = "";
		for (Pattern pattern : pattern_set.getShrunkPatterns(PatternType.All))
		{
			List<Double> input = pattern.getInput();
			List<Double> output = this.network.getPrediction(input);
			for (Double value : input)
				output_text += pattern_set.getInputInterval().unshrink(value.doubleValue()) + " ";
			output_text += "= ";
			for (Double value : output)
				output_text += pattern_set.getOutputInterval().unshrink(value.doubleValue()) + " ";
			output_text += "\n";
		}
		return output_text;
	}
}
