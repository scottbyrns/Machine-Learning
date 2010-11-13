package com.scottbyrns.ml.neural;

import java.util.Iterator;
import java.util.Vector;

/**
 * Created by scott
 * Date: Nov 11, 2010
 * Time: 5:42:42 PM
 */
public class DefaultSynapseLayer implements SynapseLayer {

    private Vector<Synapse> synapses;

    public DefaultSynapseLayer() {
        setSynapses(new Vector<Synapse>());
    }

    /**
     * Add a synapse to the layer.
     * @param synapse synapse to add
     */
    public void add(Synapse synapse) {
        getSynapses().add(synapse);
    }

    /**
     * Get the synapse at the specified index.
     * @param index
     * @return The synapse at the specified index or null if the index is out of bounds.
     */
    public Synapse getSynapseAtIndex(int index) {
        try {
            return getSynapses().get(index);
        }
        catch (ArrayIndexOutOfBoundsException e) {
            return null;
        }
    }

    /**
     * Get a weight vector representing the weights of the synapses in this layer.
     * @return Vector of weight values, null in case of error.
     */
    public Vector<Double> getWeightVector() {
        try {
			Vector<Double> list = new Vector<Double>();
			for (Synapse synapse : getSynapses()) {
                list.add(synapse.getWeight());
            }

			return list;
        }
        catch (RuntimeException e) {
            return null;
        }
    }


    /**
     * Get the size of the synapse vector in this layer.
     * @return size of the synapse vector.
     */
    public int size() {
        return getSynapses().size();
    }

    /**
     * Get the synapse vector
     * @return synapse vector
     */
    private Vector<Synapse> getSynapses() {
        return synapses;
    }

    /**
     * Set the synapse vector to the provided input vector.
     * @param synapses new synapse vector.
     */
    private void setSynapses(Vector<Synapse> synapses) {
        this.synapses = synapses;
    }

    /**
     * Get an iterator for the synapse vector.
     * @return iterator
     */
    public Iterator<Synapse> getSynapsesIterator () {
        return getSynapses().iterator();
    }
}
