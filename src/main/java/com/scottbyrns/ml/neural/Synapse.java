package com.scottbyrns.ml.neural;

import java.io.Serializable;

/**
 * Object representation of a synapse.
 * Created by scott
 * Date: Nov 10, 2010
 * Time: 5:29:01 PM
 */
public interface Synapse extends Serializable {
    
    /**
     * Set the input neuron of the synapse.
     * @param neuron
     */
    public void setInputNeuron (Neuron neuron);

    /**
     * Get the input neuron of the synapse.
     * @return
     */
    public Neuron getInputNeuron ();

    /**
     * Set the output neuron of the synapse.
     * @param neuron
     */
    public void setOutputNeuron (Neuron neuron);

    /**
     * Get the output neuron of the synapse.
     * @return
     */
    public Neuron getOutputNeuron ();

    /**
     * Get the weight of the synapse.
     * @return
     */
    public double getWeight ();

    /**
     * Set the weight of the synapse.
     * @param weight
     */
    public void setWeight (double weight);

    /**
     * Reset the weight of the synapse.
     */
    public void resetWeight ();

    /**
     * Get the value of the synapse.
     * @return
     */
    public double getValue ();

    /**
     * Set the value of the synapse.
     * @param value
     */
    public void setValue (double value);
}
