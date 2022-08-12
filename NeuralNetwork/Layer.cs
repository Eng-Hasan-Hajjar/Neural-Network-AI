using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{   
        // A layer of neurone in a neuronal network
        //             / N1 ----->        OUTPUTS
        // INPUTS ===> - N2 ----->  (1 output for each 
        //             \ Ni ----->  neuron of the layer)
        // Each neuron of the layer has the same number of
        // inputs, this is the number of inputs of the layer
        // itself.
    class Layer
    {
        #region PROTECTED FIELDS (State of the layer)

        // Number of neurons in the layer
        protected int nn;
        // Number of inputs of the layer
        protected int ni;
        // Neurons of the layer
        protected Neuron[] neurons;
        // Last output of the layer
        protected float[] output;

        #endregion

        #region PUBLIC ACCES TO LAYER STATE

        // Number of neurons in this layer
        public int N_Neurons
        {
            get { return nn; }
        }
        // Number of inputs of this layer
        public int N_Inputs
        {
            get { return ni; }
        }
        // Indexer of layer's neurons
        public Neuron this[int neurone]
        {
            get { return neurons[neurone]; }
        }
        // Return last output vector of the layer
        public float[] Last_Output
        {
            get { return output; }
        }

        #endregion

        #region LAYER CONSTRUCTORS

        // Build a new Layer with neurons neurones. Every neuron 
        // has "inputs" inputs and the activation function f.
        public Layer(int neurons, int inputs, ActivationFunction f)
        {
            nn = neurons;
            ni = inputs;
            this.neurons = new Neuron[nn];
            output = new float[nn];
            for (int i = 0; i < neurons; i++)
                this.neurons[i] = new Neuron(inputs, f);
        }

        // Build a new Layer with neurons neurones. Every neuron 
        // has "inputs" inputs and the sigmoid activation function.
        public Layer(int neurons, int inputs)
        {
            nn = neurons;
            ni = inputs;
            this.neurons = new Neuron[nn];
            output = new float[nn];
            for (int i = 0; i < neurons; i++)
                this.neurons[i] = new Neuron(inputs);
        }
        /// Set the activation function f to all neurons of the layer
        public void setActivationFunction(ActivationFunction f)
        {
            foreach (Neuron n in neurons)
                n.F = f;
        }

        #endregion

        #region INITIALIZATION FUNCTIONS

        // Randomize all neurons weights
        public void randomizeWeight()
        {
            foreach (Neuron n in neurons)
                n.randomizeWeight();
        }
        // Randomize all neurons thresholds
        public void randomizeThreshold()
        {
            foreach (Neuron n in neurons)
                n.randomizeThreshold();
        }
        // Randomize all neurons threshold and weights
        public void randomizeAll()
        {
            randomizeWeight();
            randomizeThreshold();
        }
        // Set the randomization interval for all neurons
        public void setRandomizationInterval(float min, float max)
        {
            foreach (Neuron n in neurons)
            {
                n.Randomization_Max = max;
                n.Randomization_Min = min;
            }
        }

        #endregion

        #region OUTPUT VALUE ACCES

        // Compute output of the layer.
        // The output vector contains the output of each neuron of the layer.
        public float[] Output(float[] input)
        {
            if (input.Length != ni)
               throw new Exception("LAYER : Wrong input vector size, unable to compute output value");
            for (int i = 0; i < nn; i++)
                output[i] = neurons[i].ComputeOutput(input);
            return output;
        }

        #endregion



    }
}
