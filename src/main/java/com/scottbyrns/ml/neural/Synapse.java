package com.scottbyrns.ml.neural;

import java.io.Serializable;

/**
 * Created by scott
 * Date: Nov 10, 2010
 * Time: 5:29:01 PM
 */
public interface Synapse extends Serializable {
    public void setInputNeuron (Neuron neuron);
    public Neuron getInputNeuron ();

    public void setOutputNeuron (Neuron neuron);
    public Neuron getOutputNeuron ();

    public double getWeight ();
    public void setWeight (double weight);
    public void resetWeight ();

    public double getValue ();

    public void setValue (double value);
}
