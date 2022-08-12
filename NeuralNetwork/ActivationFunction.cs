using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
  public interface ActivationFunction
    {
            // Compute function value
            float Output(float x);
            // Compute the diff of the function
            float OutputPrime(float x);
    }

    #region SIGMOID ACTIVATION FUNCTION
    // The sigmoid activation function
    /// Here is the definition of the sigmoid activation function
    ///                1
    /// f(x) = -----------------   beta > 0
    ///         1 + e^(-beta*x)
    /// f'(x) = beta * f(x) * ( 1 - f(x) )   
    public class SigmoidActivationFunction : ActivationFunction
    {
        // The beta parameter of the sigmoid
        protected float beta = 1.0f;
        // Get or set the beta parameter of the function
        // ( beta must be positive )
        public float Beta
        {
            get { return beta; }
            set { beta = (value > 0) ? value : 1.0f; }
        }
        /// Get the name of the activation function
        public string Name
        {
            get { return "Sigmoid"; }
        }
        ///                 1
        /// f(x) = -----------------   beta > 0
        ///         1 + e^(-beta*x)
        public virtual float Output(float x)
        {
            return (float)(1 / (1 + Math.Exp(-beta * x)));
        }
        // f'(x) = beta * f(x) * ( 1 - f(x) )
        public virtual float OutputPrime(float x)
        {
            float y = Output(x);
            return (beta * y * (1 - y));
        }
    }

    #endregion

    #region LINEAR ACTIVATION FUNCTION
    /// The linear activation function
    ///        |1            if x > 0.5/A
    /// f(x) = |A * x + 0.5  if 0.5/A > x > -0.5/A
    ///        |0            if -0.5/A > x
    ///             A > 0      
    public class LinearActivationFunction : ActivationFunction
    {
        // The A parameter of the linear function
        protected float a = 1.0f;
        // Usefull to compute function value
        protected float threshold = 0.5f;
        // Get or set the A parameter of the function
        // ( A must be positive )
        public float A
        {
            get { return a; }
            set
            {
                a = (value > 0) ? value : 1.0f;
                threshold = 0.5f / a;
            }
        }
        // Get the name of the activation function
        public string Name
        {
            get { return "Linear"; }
        }
        //  Get the activation function value
        public virtual float Output(float x)
        {
            if (x > threshold) return 1;
            else if (x < -threshold) return 0;
            else return a * x + 0.5f;
        }
        /// Get the diff function value
        public virtual float OutputPrime(float x)
        {
            if (x > threshold) return 0;
            else if (x < -threshold) return 0;
            else return a;
        }
    }
    #endregion

    #region HEAVISIDE ACTIVATION FUNCTION
    /// The heaviside activation function
    /// f(x) = 0 if 0>x
    /// f(x) = 1 if x>0
    public class HeavisideActivationFunction : ActivationFunction
    {
        /// Get the name of the activation function
        public string Name
        {
            get { return "Heaviside"; }
        }
        ///  Get the heaviside function value
        public virtual float Output(float x)
        {
            if (x > 0) return 1;
            else return 0;
        }
        /// Get the derivative function value
        /// Simulate an impulse at origin...
        public virtual float OutputPrime(float x)
        {
            if (Math.Abs(x) < 0.0001) return float.MaxValue;
            else return 0;
        }
    }


    #endregion

}
