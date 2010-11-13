package com.scottbyrns.ml.neural;

import com.scottbyrns.ml.neural.Synapse;

import java.util.Vector;

/**
 * Created by scott
 * Date: Nov 11, 2010
 * Time: 5:41:34 PM
 */
public interface SynapseLayer {

    /**
     * Add a synapse to the layer.
     * @param synapse
     */
    public void add (Synapse synapse);

    /**
     * Get the synapse at the specified index.
     * @param index
     * @return The synapse at the specified index or null if the index is out of bounds.
     */
    public Synapse getSynapseAtIndex(int index);

    /**
     * Get the size of the synapse vector in this layer.
     * @return
     */
    public int size();

    /**
     * Get a weight vector representing the weights of the synapses in this layer.
     * @return Vector of weight values, null in case of error.
     */
    public Vector<Double> getWeightVector();

}
