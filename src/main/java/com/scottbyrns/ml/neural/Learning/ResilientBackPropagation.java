package com.scottbyrns.ml.neural.Learning;

import com.scottbyrns.ml.datasets.Pattern;
import com.scottbyrns.ml.neural.FeedForwardNeuralNetwork;
import com.scottbyrns.ml.neural.Synapse;
import com.scottbyrns.ml.neural.SynapseLayer;

import java.util.Vector;

/**
 * Created by scott
 * Date: Nov 12, 2010
 * Time: 4:03:08 PM
 */
public class ResilientBackPropagation extends AbstractFeedForwardNetworkLearningAlgorithm {


    private double deltaNought;
    private double deltaMin;
    private double deltaMax;

    private static final double	DEFAULT_DELTA_NOUGHT = 0.1;
    private static final double	DEFAULT_DELTA_MAX = 50;
    private static final double	DEFAULT_DELTA_MIN = Math.pow(10, -6);

    private static final double	DEFAULT_ETA_MINUS = 0.5;
	private static final double	DEFAULT_ETA_PLUS = 1.2;


    @Override
    protected void trainEpoch(Vector<Pattern> patterns) {
        for (SynapseLayer synapseLayer : getNetwork().getSynapseLayers()) {
             for (int i = 0; i < synapseLayer.size(); i += 1) {
                 getDeltaWeight().put(synapseLayer.getSynapseAtIndex(i), DEFAULT_DELTA_NOUGHT);
             }
        }

        if (getCurrentEpoch() >= 0) {
            for (Pattern pattern : patterns) {
                getNetwork().feedForward(pattern.getInput());
                calculateDeltas(pattern.getOutput());

                for (SynapseLayer synapseLayer : getNetwork().getSynapseLayers()) {
                     for (int i = 0; i < synapseLayer.size(); i += 1) {
                         updatePartialDerivative(synapseLayer.getSynapseAtIndex(i));
                     }
                }

            }
        }
        updateWeights();
    }

    public ResilientBackPropagation (FeedForwardNeuralNetwork network, int maximumEpochs, double minimumError) {
        super(network, maximumEpochs, minimumError);
        initializeSynapseDeltas();
    }

    public ResilientBackPropagation (FeedForwardNeuralNetwork network) {
        super(network);
        initializeSynapseDeltas();
    }

    public ResilientBackPropagation (FeedForwardNeuralNetwork network, int maximumEpochs) {
        super(network, maximumEpochs);
        initializeSynapseDeltas();
    }

    public ResilientBackPropagation (FeedForwardNeuralNetwork network, double minimumError) {
        super(network, minimumError);
        initializeSynapseDeltas();
    }

    private void initializeSynapseDeltas () {
        setDeltaMax(DEFAULT_DELTA_MAX);
        setDeltaMin(DEFAULT_DELTA_MIN);
        setDeltaNought(DEFAULT_DELTA_NOUGHT);

        for (SynapseLayer synapseLayer : getNetwork().getSynapseLayers()) {
            for (int i = 0; i < synapseLayer.size(); i += 1) {
                setWeightUpdate(synapseLayer.getSynapseAtIndex(i), getDeltaNought());
            }
        }
    }

	/**
	 * Updates the outgoing synapses' weights to improve the network's output
	 */
	private void updateWeights() {

        Synapse synapse;

        for (SynapseLayer synapseLayer : getNetwork().getSynapseLayers()) {
             for (int i = 0; i < synapseLayer.size(); i += 1) {

                synapse = synapseLayer.getSynapseAtIndex(i);

                double u_delta_weight = getDeltaWeight().get(synapse);
                double u_error_partial_derivative = getErrorPartialDerivative().get(synapse);
                double u_old_error_partial_derivative = getOldErrorPartialDerivative().get(synapse);
                double u_weight_update = getWeightUpdate().get(synapse);
                if (partialDerivativeMaintainsSign(synapse))
                {
                    getDeltaWeight().put(synapse, (Math.min(u_delta_weight * DEFAULT_ETA_PLUS, getDeltaMax())));
                    getWeightUpdate().put(synapse, (-Math.signum(u_error_partial_derivative) * getDeltaWeight().get(synapse)));
                    synapse.setWeight(synapse.getWeight() + getWeightUpdate().get(synapse));
                    getOldErrorPartialDerivative().put(synapse, u_error_partial_derivative);
                }
                else if (partialDerivativeChangesSign(synapse))
                {
                    getDeltaWeight().put(synapse, (Math.max(u_delta_weight * DEFAULT_ETA_MINUS, getDeltaMin())));
                    getOldErrorPartialDerivative().put(synapse, 0.0);
                }
                else
                {
                    getWeightUpdate().put(synapse, (-Math.signum(u_error_partial_derivative) * u_delta_weight));
                    synapse.setWeight(synapse.getValue() + getWeightUpdate().get(synapse));
                    getOldErrorPartialDerivative().put(synapse, (u_error_partial_derivative));
                }

             }
        }
    }


	/**
	 * Updates the partial derivative for the specified synapse
	 *
	 * @param synapse The synapse where the partial derivative is going to be updated
	 */
	private void updatePartialDerivative(Synapse synapse) {
		double derivative = getErrorPartialDerivative().get(synapse);
		double derivativeUpdate = calculateErrorPartialDerivative(synapse);
		getErrorPartialDerivative().put(synapse, derivative + derivativeUpdate);
	}

	/**
	 * Indicates if the partial derivative changed sign in relation to the
	 * previous epoch
	 *
	 * @param synapse
	 *            The synapse that's being tested
	 * @return Returns boolean indicating whether the partial derivative changed
	 *         sign in relation to the previous epoch
	 */
	private boolean partialDerivativeChangesSign(Synapse synapse) {
		double old = 0.0;
		if (getOldErrorPartialDerivative().get(synapse) != null) {
            old = getOldErrorPartialDerivative().get(synapse);
        }

		double current = getErrorPartialDerivative().get(synapse);
		return current * old < 0.0;
	}

	/**
	 * Indicates if the partial derivative maintained sign in relation to the
	 * previous epoch
	 *
	 * @param synapse
	 *            The synapse that's being tested
	 * @return Returns boolean indicating whether the partial derivative
	 *         maintained sign in relation to the previous epoch
	 */
	private boolean partialDerivativeMaintainsSign(Synapse synapse) {
		double old = 0.0;
		if (getOldErrorPartialDerivative().get(synapse) != null) {
            old = getOldErrorPartialDerivative().get(synapse);
        }

		double current = getErrorPartialDerivative().get(synapse);
		return current * old > 0.0;
	}

    private double getDeltaNought() {
        return deltaNought;
    }

    private void setDeltaNought(double deltaNought) {
        this.deltaNought = deltaNought;
    }

    private double getDeltaMin() {
        return deltaMin;
    }

    private void setDeltaMin(double deltaMin) {
        this.deltaMin = deltaMin;
    }

    private double getDeltaMax() {
        return deltaMax;
    }

    private void setDeltaMax(double deltaMax) {
        this.deltaMax = deltaMax;
    }
}
