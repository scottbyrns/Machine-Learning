package com.scottbyrns.ml.neural;

/**
 * Type safe list of neuron types.
 *
 * @author Scott Byrns
 * Date: Nov 10, 2010
 * Time: 5:34:18 PM
 *
 * @version 1.0
 */
public enum NeuronType {
    /**
     * Node that behaves normally.
     */
    Normal,
    /**
     * Node that remains constant to provide a DC offset to each sigmoid
     */
    Bias
}
