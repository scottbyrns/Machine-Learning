package com.scottbyrns.ml.genetic;

/**
 * Created by scott
 * Date: Nov 10, 2010
 * Time: 12:03:53 PM
 */
public class MockCandidateModel implements PopulationModel {
        private int PopulationSize = 5;
        private int Generations = 500;
        private double FitnessCutoff = 1.0;
        private float PopulationCutoff = 0.25f;
        private int GenomeLength = 24;
        private int GenomeCrossLength = 8;
        private boolean UniformCross = true;
        private float MutateChance = 0.02f;
        private float MutateAmount = 0.5f;

        public int getPopulationSize () {
            return PopulationSize;
        }
        public int getGenerations () {
            return Generations;
        }
        public double getFitnessCutoff () {
            return FitnessCutoff;
        }
        public float getPopulationCutoff () {
            return PopulationCutoff;
        }
        public int getGenomeLength () {
            return GenomeLength;
        }
        public int getGenomeCrossLength () {
            return GenomeCrossLength;
        }
        public boolean isUniformCross () {
            return UniformCross;
        }
        public float getMutateChance () {
            return MutateChance;
        }
        public float getMutateAmount () {
            return MutateAmount;
        }
}
