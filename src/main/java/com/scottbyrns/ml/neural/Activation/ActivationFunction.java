package com.scottbyrns.ml.neural.Activation;

/**
 * Controls the amplitude of the output of the neuron.
 * An acceptable range of output is usually between 0 and 1, or -1 and 1.
 *
 * @author Scott Byrns
 * Date: Nov 11, 2010
 * Time: 11:25:54 AM
 *
 * @version 1.0
 */
public interface ActivationFunction {

	public double calculate(double value);

	public double calculateDerivative(double value);

}
