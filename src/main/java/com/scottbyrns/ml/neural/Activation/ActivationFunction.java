package com.scottbyrns.ml.neural.Activation;

/**
 * Created by scott
 * Date: Nov 11, 2010
 * Time: 11:25:54 AM
 */
public interface ActivationFunction {

	public double calculate(double value);

	public double calculateDerivate(double value);

}
