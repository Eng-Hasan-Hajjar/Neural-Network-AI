using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections;

namespace NeuralNetwork
{   
    // The abstract class describing a learning
    // algorithm for a neural network
    abstract class   LearningAlgorithm
    {
        #region PROTECTED FIELDS

        // The neural network
        protected NeuralNetwork nn;
        // Under this threshold value, learning will be
        // considered as complete
        protected float ERROR_THRESHOLD = 0.001f;
        // Max number of iteration to learn data
        protected int MAX_ITER = 1000;
        // Input matrix of data to learn
        protected float[][] ins;
         // output matrix of data to learn
        protected float[][] outs;
        /// Number of learning iterations done
        protected int iter = 0;
        /// Last sum of square errors computed
        protected float error = -1;

        #endregion

        #region PUBLIC ACCES TO LEARNING ALGORITHM STATE
        // Get the neural network of the learning algorithm
        public NeuralNetwork N_Network
        {
            get { return nn; }
        }
        // Get the last square error
        public float Error
        {
            get { return error; }
        }
        // Get or set the maximum sum of square errors value ( >0)
        public float ErrorTreshold
        {
            get { return ERROR_THRESHOLD; }
            set { ERROR_THRESHOLD = (value > 0) ? value : ERROR_THRESHOLD; }
        }
        // Get the current number of learning iterations done
        public int Iteration
        {
            get { return iter; }
        }
        // Get or set the maximum number of learning iterations.
        public int MaxIteration
        {
            get { return MAX_ITER; }
            set { MAX_ITER = (value > 0) ? value : MAX_ITER; }
        }

        #endregion

        #region CONSTRICTOR AND METHODS

        // Learning algorithm constructor
        public LearningAlgorithm(NeuralNetwork n)
        {
            nn = n;
        }
        // To train the neuronal network on data.
        // inputs[n] represents an input vector of 
        // the neural network and expected_outputs[n]
        // the expected ouput for this vector. 
        public virtual void Learn(float[][] inputs, float[][] expected_outputs)
        {
            if (inputs.Length < 1)
                throw new Exception("LearningAlgorithme : no input data : cannot learn from nothing");
            if (expected_outputs.Length < 1)
                throw new Exception("LearningAlgorithme : no output data : cannot learn from nothing");
            if (inputs.Length != expected_outputs.Length)
                throw new Exception("LearningAlgorithme : inputs and outputs size does not match : learning aborded ");
            ins = inputs;
            outs = expected_outputs;
        }
        #endregion

    }

    #region BackPropagationLearningAlgorithm

    // Implementation of stockastic gradient backpropagation
    // learning algorithm
    //                     PROPAGATION WAY IN NN
    //                   ------------------------->
    //        o ----- Sj = f(WSj) ----> o ----- Si = f(WSi) ----> o
    //      Neuron j                Neuron i                   Neuron k
    //    (layer L-1)               (layer L)                 (layer L+1)
    // 
    // For the neuron i :
    // -------------------
    // W[i,j](n+1) = W[i,j](n) + alpha * Ai * Sj + gamma * ( W[i,j](n) - W[i,j](n-1) )
    // T[i](n+1) = T[i](n) - alpha * Ai + gamma * ( T[i](n) - T[i](n-1) )
    // 
    //		with :
    //				Ai = f'(WSi) * (expected_output_i - si) for output layer
    //				Ai = f'(WSi) * SUM( Ak * W[k,i] )       for others
    // NOTE : This is stockastic version of the algorithm because the error
    // is back-propaged after every learning case. There is another version
    // of this algorithm which works on global error.
    class BackPropagationLearningAlgorithm : LearningAlgorithm
    {

        #region PRETECTED FIELDS
        // the alpha parameter of the algorithm
        protected float alpha = 0.5f;
        // the gamma parameter of the algorithm
        protected float gamma = 0.2f;
        // The error vector
        protected float[] e;

        #endregion

        #region PUBLIC ACCES TO PARAMETERS OF ALGORITHM

        // get or set the alpha parameter of the algorithm
        // between 0 and 1, must be >0
        public float Alpha
        {
            get { return alpha; }
            set { alpha = (value > 0) ? value : alpha; }
        }
        // get or set the gamma parameter of the algorithm
        // (Rumelhart coef)
        // between 0 and 1.
        public float Gamma
        {
            get { return gamma; }
            set { gamma = (value > 0) ? value : gamma; }
        }

        #endregion

        #region CONSTRUCTOR
        // Build a new BackPropagation learning algorithm instance
        // with alpha = 0,5 and gamma = 0,3
        public BackPropagationLearningAlgorithm(NeuralNetwork nn) : base(nn)
        {
        }

        #endregion

        #region LEARNING METHODS

        // To train the neuronal network on data.
        // inputs[n] represents an input vector of 
        // the neural network and expected_outputs[n]
        // the expected ouput for this vector. 
        public override void Learn(float[][] inputs, float[][] expected_outputs)
        {
            base.Learn(inputs, expected_outputs);
            float[] nout;
            float err;
            iter = 0;
            do
            {
                error = 0f;
                e = new float[nn.N_Outputs];
                for (int i = 0; i < ins.Length; i++)
                {
                    err = 0f;
                    nout = nn.Output(inputs[i]);
                    for (int j = 0; j < nout.Length; j++)
                    {
                        e[j] = outs[i][j] - nout[j];
                        err += e[j] * e[j];
                    }
                    err /= 2f;
                    error += err;
                    ComputeA(i);
                    setWeight(i);
                }
                iter++;
            }
            while (iter < MAX_ITER && this.error > ERROR_THRESHOLD);
        }
        // Compute the "A" parameter for each neuron
        // of the network
        protected void ComputeA(int i)
        {
            float sk;
            int l = nn.N_Layers - 1;
            // For the last layer
            for (int j = 0; j < nn[l].N_Neurons; j++)
                nn[l][j].A = nn[l][j].OutputPrime * e[j];
            // For other layer
            for (l--; l >= 0; l--)
            {
                for (int j = 0; j < nn[l].N_Neurons; j++)
                {
                    sk = 0f;
                    for (int k = 0; k < nn[l + 1].N_Neurons; k++)
                        sk += nn[l + 1][k].A * nn[l + 1][k][j];
                    nn[l][j].A = nn[l][j].OutputPrime * sk;
                }
            }
        }
        // Set new neron's weights
        protected void setWeight(int i)
        {
            float[] lin;
            for (int j = 0; j < nn.N_Layers; j++)
            {
                if (j == 0) lin = ins[i];
                else lin = nn[j - 1].Last_Output;
                for (int n = 0; n < nn[j].N_Neurons; n++)
                {
                    for (int k = 0; k < nn[j][n].N_Inputs; k++)
                        nn[j][n][k] += alpha * lin[k] * nn[j][n].A + gamma * (nn[j][n][k] - nn[j][n].Last_W[k]);
                    nn[j][n].Threshold -= alpha * nn[j][n].A + gamma * (nn[j][n].Threshold - nn[j][n].Last_Threshold);
                }
            }
        }

        #endregion



    }
    #endregion

}
