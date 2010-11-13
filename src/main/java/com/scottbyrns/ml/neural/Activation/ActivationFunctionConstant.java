package com.scottbyrns.ml.neural.Activation;

/**
 * Created by scott
 * Date: Nov 11, 2010
 * Time: 12:56:10 PM
 */
public class ActivationFunctionConstant implements ActivationFunction {
	/**
	 * Returns the activation function's result
	 *
	 * @param value
	 *            This value is unused since the constant function always
	 *            returns the same value
	 * @return Returns 1.0
	 */
	public double calculate(double value) {
		value++; // FIXME Replace with appropriate SuppressWarning
		return 1;
	}

	/**
	 * Returns the activation function's derivative's result
	 *
	 * @param value
	 *            This value is unused since the constant function always
	 *            returns the same value
	 * @return Returns 0.0
	 */
	public double calculateDerivate(double value) {
		value++; // FIXME Replace with appropriate SuppressWarning
		return 0;
	}
}
