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
     * @param neuron input neuron of the synapse
     */
    public void setInputNeuron (Neuron neuron);

    /**
     * Get the input neuron of the synapse.
     * @return input neuron of the synapse
     */
    public Neuron getInputNeuron ();


    /**
     * Set the output neuron of the synapse.
     * @param neuron output neuron of the synapse
     */
    public void setOutputNeuron (Neuron neuron);

    /**
     * Get the output neuron of the synapse.
     * @return output neuron of the synapse
     */
    public Neuron getOutputNeuron ();


    /**
     * Get the weight of the synapse.
     * @return weight of the synapse
     */
    public double getWeight ();

    /**
     * Set the weight of the synapse.
     * @param weight of the synapse
     */
    public void setWeight (double weight);

    /**
     * Reset the weight of the synapse.
     */
    public void resetWeight ();

    
    /**
     * Get the value of the synapse.
     * @return value of the synapse
     */
    public double getValue ();

    /**
     * Set the value of the synapse.
     * @param value of the synapse
     */
    public void setValue (double value);
}
