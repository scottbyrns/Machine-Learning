package com.scottbyrns.ml.neural.Activation;

import com.scottbyrns.ml.Mathematics;

/**
 * Created by scott
 * Date: Nov 11, 2010
 * Time: 12:53:55 PM
 */
public class ActivationFunctionSigmoid implements ActivationFunction {
    /**
	 * Returns the sigmoid's output for the given value
	 *
	 * @param value
	 *            The input for the sigmoid function
	 * @return Returns the sigmoid's output
	 */
	public double calculate(double value) {
		return Mathematics.sigmoid(value);
	}

	/**
	 * Returns the sigmoid's derivative's output for the given value
	 *
	 * @param value
	 *            The input for the derivated sigmoid function
	 * @return Returns the derivated sigmoid's output
	 */
	public double calculateDerivate(double value) {
		return calculate(value) * (1 - calculate(value));
	}

    
}
