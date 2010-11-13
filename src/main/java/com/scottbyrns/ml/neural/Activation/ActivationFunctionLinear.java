package com.scottbyrns.ml.neural.Activation;

/**
 * Created by scott
 * Date: Nov 11, 2010
 * Time: 12:54:59 PM
 */
public class ActivationFunctionLinear implements ActivationFunction {
	/**
	 * Returns the activation function's output
	 * @param value The input value for the activation function
	 * @return Returns the same value that was passed as a parameter
	 */
	public double calculate(double value) {
		return value;
	}

	/**
	 * Returns the activation function's derivative's output
	 * @param value This value is unused since the output of the derivative of a linear function is always the same
	 * @return Returns 1.0
	 */
	public double calculateDerivate(double value) {
		value++; // FIXME Replace with appropriate SuppressWarning
		return 1;
	}
}
