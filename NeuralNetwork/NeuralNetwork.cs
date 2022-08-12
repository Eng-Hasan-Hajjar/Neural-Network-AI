using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;

namespace NeuralNetwork
{
    class NeuralNetwork
    {
        // Implementation of artificial neural network
        //                        o
        //                        o  o  o  
        //    INPUT VECTOR =====> o  o  o =====> OUTPUT VECTOR
        //                        o  o  o  
        //                        o
        //                      NERON LAYERS
        // Each neuron of the layer N-1 is conected to 
        // every neuron of the layer N.
        // At the begining the neural network needs to
        // learn using couples (INPUT, EXPECTED OUTPUT)
        // and a learnig algorithm.

        #region PROTECTED FIELDS (STATE OF THE NETWORK)

        /// Layers of neuron in the network
        protected Layer[] layers;
        /// Number of inputs of the network
        /// (number of inputs of the first layer)
        protected int ni;
        /// Learning algorithm used by the network
        protected LearningAlgorithm la;

        #endregion

        #region PUBLIC ACESS TO NETWORK STATE

        /// Get number of inputs of the network
        /// (network input vector size)
        public int N_Inputs
        {
            get { return ni; }
        }
        /// Get number of output of the network
        /// (network output vector size)
        public int N_Outputs
        {
            get { return layers[N_Layers - 1].N_Neurons; }
        }
        /// Get number of inputs of the network
        /// (network input vector size)
        public int N_Layers
        {
            get { return layers.Length; }
        }
        /// Get or set network learning algorithm
        public LearningAlgorithm LearningAlg
        {
            get { return la; }
            set { la = (value != null) ? value : la; }
        }
        /// Get the n th Layer of the network 
        public Layer this[int n]
        {
            get { return layers[n]; }
        }

        #endregion

        #region NEURAL NETWORK CONSTRUCTOR

        // Create a new neural network
        // with "inputs" inputs and size of "layers"
        // layers of neurones.
        // The layer i is made with layers_desc[i] neurones.
        // The activation function of each neuron is set to n_act.
        // The lerning algorithm is set to learn.
        public NeuralNetwork(int inputs, int[] layers_desc, ActivationFunction n_act, LearningAlgorithm learn)
        {
            if (layers_desc.Length < 1)
                throw new Exception("PERCEPTRON : cannot build perceptron, it must have at least 1 layer of neurone");
            if (inputs < 1)
                throw new Exception("PERCEPTRON : cannot build perceptron, it must have at least 1 input");
            la = learn;
            ni = inputs;
            layers = new Layer[layers_desc.Length];
            layers[0] = new Layer(layers_desc[0], ni);
            for (int i = 1; i < layers_desc.Length; i++)
                layers[i] = new Layer(layers_desc[i], layers_desc[i - 1], n_act);
        }
        // Create a new neural network
        // with "inputs" inputs and size of "layers"
        // layers of neurones.
        // The layer i is made with layers_desc[i] neurones.
        // The activation function of each neuron is set to n_act.
        // The lerning algorithm is set to default (Back Propagation).
        public NeuralNetwork(int inputs, int[] layers_desc, ActivationFunction n_act)
        {
            if (layers_desc.Length < 1)
                throw new Exception("PERCEPTRON : cannot build perceptron, it must have at least 1 layer of neurone");
            if (inputs < 1)
                throw new Exception("PERCEPTRON : cannot build perceptron, it must have at least 1 input");
            la = new BackPropagationLearningAlgorithm(this);
            ni = inputs;
            layers = new Layer[layers_desc.Length];
            layers[0] = new Layer(layers_desc[0], ni);
            for (int i = 1; i < layers_desc.Length; i++)
                layers[i] = new Layer(layers_desc[i], layers_desc[i - 1], n_act);
        }
        // Create a new neural network
        // with "inputs" inputs and size of "layers"
        // layers of neurones.
        // The layer i is made with layers_desc[i] neurones.
        // The activation function of each neuron is set to default (Sigmoid with beta = 1).
        // The lerning algorithm is set to default (Back Propagation).
        public NeuralNetwork(int inputs, int[] layers_desc)
        {
            if (layers_desc.Length < 1)
                throw new Exception("PERCEPTRON : cannot build perceptron, it must have at least 1 layer of neurone");
            if (inputs < 1)
                throw new Exception("PERCEPTRON : cannot build perceptron, it must have at least 1 input");
            la = new BackPropagationLearningAlgorithm(this);
            ni = inputs;
            ActivationFunction n_act = new SigmoidActivationFunction();
            layers = new Layer[layers_desc.Length];
            layers[0] = new Layer(layers_desc[0], ni);
            for (int i = 1; i < layers_desc.Length; i++)
                layers[i] = new Layer(layers_desc[i], layers_desc[i - 1], n_act);
        }

        #endregion

        #region INITIALIZATION FUNCTIONS

        // Randomize all neurones weights between -0.5 and 0.5
        public void randomizeWeight()
        {
            foreach (Layer layerr in layers)
                layerr.randomizeWeight();
        }
        // Randomize all neurones threholds between 0 and 1
        public void randomizeThreshold()
        {
            foreach (Layer layerr in layers)
                layerr.randomizeThreshold();
        }
        // Randomize all neurones threholds between 0 and 1
        // and weights between -0.5 and 0.5
        public void randomizeAll()
        {
            foreach (Layer layerr in layers)
                layerr.randomizeAll();
        }
        // Set an activation function to all neurons of the network
        public void setActivationFunction(ActivationFunction f)
        {
            foreach (Layer layerr in layers)
                layerr.setActivationFunction(f);
        }
        // Set the interval in which weights and threshold will be randomized
        public void setRandomizationInterval(float min, float max)
        {
            foreach (Layer layerr in layers)
                layerr.setRandomizationInterval(min, max);
        }

        #endregion

        #region OUPUT METHODS

        // Compute the value for the specified input
        public float[] Output(float[] input)
        {
            if (input.Length != ni)
                throw new Exception("PERCEPTRON : Wrong input vector size, unable to compute output value");
            float[] result;
            result = layers[0].Output(input);
            for (int i = 1; i < N_Layers; i++)
                result = layers[i].Output(result);
            return result;
        }



        #endregion

        #region PERSISTANCE IMPLEMENTATION
        // Save the Neural Network in a binary formated file
        public void save(string file)
        {
            IFormatter binFmt = new BinaryFormatter();
            Stream s = File.Open(file, FileMode.Create);
            binFmt.Serialize(s, this);
            s.Close();
        }
        // Load a neural network from a binary formated file
        public static NeuralNetwork load(string file)
        {
            NeuralNetwork result;
            try
            {
                IFormatter binFmt = new BinaryFormatter();
                Stream s = File.Open(file, FileMode.Open);
                result = (NeuralNetwork)binFmt.Deserialize(s);
                s.Close();
            }
            catch (Exception e)
            {
                throw new Exception("NeuralNetwork : Unable to load file " + file + " : " + e);
            }
            return result;
        }
        #endregion

    }
}
