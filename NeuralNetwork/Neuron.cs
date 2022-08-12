using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{    //  Mohammad      ------          Ali     ------            Hasan
     // Class representing an artificial neuron
     //  --------------> * W[0] \                              -----  
     //  --------------> * W[1] - + -------> -threshold -------| f | ---------> O
     //
     //  --------------> * W[i] /                              -----
     //     SYNAPSES      WEIGHT             THRESHOLD       ACTIVATION       OUTPUT
    class Neuron
    {
        #region PROTECTED FIELDS (State variables)
        // Pseudo random number generator to initialize neuron weight
		protected static Random rand = new Random();
        // Minimum value for randomisation of weights and threshold
		protected float R_MIN = -1;
        // Maximum value for randomization of weights and threshold
		protected float R_MAX = 1;
        // Weight of every synapse
        protected float[] w;
        // Last weight of every synapse
        protected float[] last_w;
        // Threshold of the neuron
        protected float threshold = 0f;
        // Last threshold of the neuron
        protected float last_threshold = 0f;
        // Activation function of the neuron
        protected ActivationFunction f = null;
        // Value of the last neuron ouput
        protected float o = 0f;
        // Last value of synapse sum minus threshold
        protected float ws = 0f;
        /// Usefull for backpropagation algorithm
        protected float a;
        #endregion

        #region PUBLIC ACCES TO STATE OF THE NEURON

        //  Number of neuron inputs (synapses)
        public int N_Inputs
        {
            get { return w.Length; }
        }
        // Indexer of the neuron to get or set weight of synapses
        public float this[int synapse]
        {
            get { return w[synapse]; }
            set { last_w[synapse] = w[synapse]; w[synapse] = value; }
        }
        // To get or set the threshold value of the neuron
        public float Threshold
        {
            get { return threshold; }
            set { last_threshold = threshold; threshold = value; }
        }
        // Get the last output of the neuron
        public float Output
        {
            get { return o; }
        }
        // Get the last output prime of the neuron (f'(ws))
        public float OutputPrime
        {
            get { return f.OutputPrime(ws); }
        }
// Get the last sum of inputs
public float WS
        {
            get { return ws; }
        }
         // Get or set the neuron activation function
      public ActivationFunction F
        {
            get { return f; }
            set { f = value; }
        }
        // Get or set a value of the neuron
        // (usefull for backpropagation learning algorithm)
        public float A
        {
            get { return a; }
            set { a = value; }
        }
        // Get the last threshold value of the neuron
        public float Last_Threshold
        {
            get { return last_threshold; }
        }
        // Get the last weights of the neuron
        public float[] Last_W
        {
            get { return last_w; }
        }
        // Get or set the minimum value for randomisation of weights and threshold
        public float Randomization_Min
        {
            get { return R_MIN; }
            set { R_MIN = value; }
        }
        // Get or set the maximum value for randomization of weights and threshold
        public float Randomization_Max
        {
            get { return R_MAX; }
            set { R_MAX = value; }
        }
        #endregion

        #region NEURON CONSTRUCTOR

        // Build a neurone with Ni inputs
        public Neuron(int Ni, ActivationFunction af)
        {
            w = new float[Ni];
            last_w = new float[Ni];
            f = af;
        }
        /// Build a neurone with Ni inputs whith a default 
        /// activation function (SIGMOID)
        public Neuron(int Ni)
        {
            w = new float[Ni];
            last_w = new float[Ni];
            f = new SigmoidActivationFunction();
        }

        #endregion

        #region PUBLIC METHODS (INITIALIZATION FUNCTIONS)
        // Randomize Weight for each input between R_MIN and R_MAX
        public void randomizeWeight()
        {
            for (int i = 0; i < N_Inputs; i++)
            {
                w[i] = R_MIN + (((float)(rand.Next(1000))) / 1000f) * (R_MAX - R_MIN);
                last_w[i] = 0f;
            }
        }
        // Randomize the threshold (between R_MIN and R_MAX)
		public void randomizeThreshold()
        {
            threshold = R_MIN + (((float)(rand.Next(1000))) / 1000f) * (R_MAX - R_MIN);
        }
        // Randomize the threshold and the weights
        public void randomizeAll()
        {
            randomizeWeight();
            randomizeThreshold();
        }

        #endregion

        #region PUBLIC METHODS (COMPUTE THE OUTPUT VALUE)
        // Compute the output of the neurone
        public float ComputeOutput(float[] input)
        {
            if (input.Length != N_Inputs)
                throw new Exception("NEURONE : Wrong input vector size, unable to compute output value");
               ws = 0;
            for (int i = 0; i < N_Inputs; i++)
                ws += w[i] * input[i];
            ws -= threshold;
            if (f != null)
                o = f.Output(ws);
            else
                o = ws;
            return o;
        }
        #endregion


    }
}