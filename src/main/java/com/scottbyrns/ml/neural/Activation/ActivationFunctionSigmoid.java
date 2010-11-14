package com.scottbyrns.ml.neural.Activation;

import com.scottbyrns.ml.Mathematics;

/**
 * Controls the amplitude of the output of the neuron.
 * An acceptable range of output is usually between 0 and 1, or -1 and 1.
 *
 * @author Scott Byrns
 * Date: Nov 11, 2010
 * Time: 12:53:55 PM
 *
 * @version 1.0
 */
public class ActivationFunctionSigmoid implements ActivationFunction {
    /**
	 * Returns the sigmoid's output for the given value
	 *
	 * @param value The input for the sigmoid function
	 * @return Returns the sigmoid's output
	 */
	public double calculate(double value) {
		return Mathematics.sigmoid(value);
	}

	/**
	 * Returns the sigmoid's derivative's output for the given value
	 *
	 * @param value The input for the derivated sigmoid function
	 * @return Returns the derivated sigmoid's output
	 */
	public double calculateDerivative(double value) {
		return calculate(value) * (1 - calculate(value));
	}

    
}
