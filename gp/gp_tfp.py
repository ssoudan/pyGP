import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pylab as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#
# see https://github.com/tensorflow/probability/blob/master/tensorflow_probability/g3doc/api_docs/python/tfp/distributions/GaussianProcessRegressionModel.md
#
tfd = tfp.distributions
psd_kernels = tfp.positive_semidefinite_kernels


def evalMLE(X, Y, x):  # type: (Any, Any, Any) -> Tuple[Any, Union[Union[List[Optional[Any]], Tuple[Optional[Any], ...], None], Any]]

    # Define a kernel with trainable parameters. Note we transform the trainable
    # variables to apply a positivity constraint.
    amplitude = tf.exp(tf.Variable(np.float64(0)), name='amplitude')
    length_scale = tf.exp(tf.Variable(np.float64(0)), name='length_scale')
    kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)

    observation_noise_variance = tf.exp(
        tf.Variable(np.float64(-5)), name='observation_noise_variance')

    # We'll use an unconditioned GP to train the kernel parameters.
    gp = tfd.GaussianProcess(
        kernel=kernel,
        index_points=X,
        observation_noise_variance=observation_noise_variance)
    neg_log_likelihood = -gp.log_prob(Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=.05, beta1=.5, beta2=.99)
    optimize = optimizer.minimize(neg_log_likelihood)

    # We can construct the posterior at a new set of `index_points` using the same
    # kernel (with the same parameters, which we'll optimize below).
    gprm = tfd.GaussianProcessRegressionModel(
        kernel=kernel,
        index_points=x,
        observation_index_points=X,
        observations=Y,
        observation_noise_variance=observation_noise_variance)

    samples = gprm.sample(10)
    # ==> 10 independently drawn, joint samples at `index_points`.

    # Now execute the above ops in a Session, first training the model
    # parameters, then drawing and plotting posterior samples.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(1000):
            _, neg_log_likelihood_ = sess.run([optimize, neg_log_likelihood])
            if i % 100 == 0:
                print("Step {}: NLL = {}".format(i, neg_log_likelihood_))

        print("Final NLL = {}".format(neg_log_likelihood_))
        samples_ = sess.run(samples)

    return x, samples_


def evalHMC(X, Y, x):  # type: (Any, Any, Any) -> Tuple[Any, Any]

    def joint_log_prob(
        index_points, observations, amplitude, length_scale, noise_variance):

      # Hyperparameter Distributions.
      rv_amplitude = tfd.LogNormal(np.float64(0.), np.float64(1))
      rv_length_scale = tfd.LogNormal(np.float64(0.), np.float64(1))
      rv_noise_variance = tfd.LogNormal(np.float64(0.), np.float64(1))

      gp = tfd.GaussianProcess(
          kernel=psd_kernels.ExponentiatedQuadratic(amplitude, length_scale),
          index_points=index_points,
          observation_noise_variance=noise_variance)

      return (
          rv_amplitude.log_prob(amplitude) +
          rv_length_scale.log_prob(length_scale) +
          rv_noise_variance.log_prob(noise_variance) +
          gp.log_prob(observations)
      )

    initial_chain_states = [
        1e-1 * tf.ones([], dtype=np.float64, name='init_amplitude'),
        1e-1 * tf.ones([], dtype=np.float64, name='init_length_scale'),
        1e-1 * tf.ones([], dtype=np.float64, name='init_obs_noise_variance')
    ]

    # Since HMC operates over unconstrained space, we need to transform the
    # samples so they live in real-space.
    unconstraining_bijectors = [
        tfp.bijectors.Softplus(),
        tfp.bijectors.Softplus(),
        tfp.bijectors.Softplus(),
    ]

    def unnormalized_log_posterior(amplitude, length_scale, noise_variance):
      return joint_log_prob(
          X, Y, amplitude, length_scale,
          noise_variance)

    num_results = 10
    [
        amplitudes,
        length_scales,
        observation_noise_variances
    ], kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=500,
        num_steps_between_results=3,
        current_state=initial_chain_states,
        kernel=tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=unnormalized_log_posterior,
                step_size=[np.float64(.15)],
                num_leapfrog_steps=3),
            bijector=unconstraining_bijectors))

    # Now we can sample from the posterior predictive distribution at a new set
    # of index points.
    gprm = tfd.GaussianProcessRegressionModel(
        # Batch of `num_results` kernels parameterized by the MCMC samples.
        kernel=psd_kernels.ExponentiatedQuadratic(amplitudes, length_scales),
        index_points=x,
        observation_index_points=X,
        observations=Y,
        # We reshape this to align batch dimensions.
        observation_noise_variance=observation_noise_variances[..., np.newaxis])
    samples = gprm.sample()

    with tf.Session() as sess:
        kernel_results_, samples_ = sess.run([kernel_results, samples])

        print("Acceptance rate: {}".format(
            np.mean(kernel_results_.inner_results.is_accepted)))

    return x, samples_


def plot(X, Y, x, y, f=None, title=None, output=None):
    # Plot posterior samples and their mean, target function, and observations.
    num_results = y.shape[0]

    plt.figure()
    plt.plot(np.stack([x[:, 0]] * num_results).T,
             y.T,
             c='r',
             alpha=.1)
    plt.plot(x[:, 0], np.mean(y, axis=0), c='k')
    if f is not None:
        plt.plot(x[:, 0], f(x))
    plt.scatter(X[:, 0], Y)
    if title is not None:
        plt.title(title)
    if output is not None:
        plt.savefig(output)
    plt.show()

