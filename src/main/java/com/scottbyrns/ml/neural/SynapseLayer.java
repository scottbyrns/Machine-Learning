package com.scottbyrns.ml.neural;

import java.util.Iterator;
import java.util.Vector;

/**
 * Weight Matrix
 * Created by scott
 * Date: Nov 11, 2010
 * Time: 5:41:34 PM
 */
public interface SynapseLayer {

    /**
     * Add a synapse to the layer.
     *
     * @param synapse
     */
    public void addSynapse(Synapse synapse);

    /**
     * Get a weight vector representing the weights of the synapses in this layer.
     *
     * @return Vector of weight values, null in case of error.
     */
    public Vector<Double> getWeightVector();

    /**
     * Get an iterator for the synapse vector.
     *
     * @return
     */
    public Iterator<Synapse> getSynapsesIterator ();

    /**
     * Set the weights of the synapses in this layer to the next values
     * of the provided weightVectorIterator
     * 
     * @param weightVectorIterator
     */
    public void setWeightVector (Iterator<Double> weightVectorIterator);

}
