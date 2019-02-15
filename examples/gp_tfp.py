import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pylab as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import examples.data as data

#
# see https://github.com/tensorflow/probability/blob/master/tensorflow_probability/g3doc/api_docs/python/tfp/distributions/GaussianProcessRegressionModel.md
#
tfd = tfp.distributions
psd_kernels = tfp.positive_semidefinite_kernels


def evalGPRSample():  # type: () -> None

    # Generate noisy observations from a known function at some random points.
    observation_noise_variance = .5
    f = lambda x: np.sin(10*x[..., 0]) * np.exp(-x[..., 0]**2)
    observation_index_points = np.random.uniform(-1., 1., 50)[..., np.newaxis]
    observations = f(observation_index_points) + np.random.normal(0., np.sqrt(observation_noise_variance))
    index_points = np.linspace(-1., 1., 100)[..., np.newaxis]

    kernel = psd_kernels.MaternFiveHalves()

    gprm = tfd.GaussianProcessRegressionModel(
        kernel=kernel,
        index_points=index_points,
        observation_index_points=observation_index_points,
        observations=observations,
        observation_noise_variance=observation_noise_variance)

    samples = gprm.sample(10)
    print(samples)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        samples_ = sess.run(samples)
        print(samples_)


def evalMLE(X, Y, x, f = None):  # type: (Any, Any, Any, Any) -> None

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

        plt.scatter(np.squeeze(X), Y)
        plt.plot(np.stack([x[:, 0]] * 10).T, samples_.T, c='r', alpha=.2)
        if f is not None:
            plt.plot(x[:, 0], f(x))
        plt.title("MLE")
        plt.show()


def evalHMC(X, Y, x, f = None):  # type: (Any, Any, Any, Any) -> None

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

    num_results = 200
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
            inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
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

        # Plot posterior samples and their mean, target function, and observations.
        plt.figure()
        plt.plot(np.stack([x[:, 0]] * num_results).T,
                 samples_.T,
                 c='r',
                 alpha=.01)
        plt.plot(x[:, 0], np.mean(samples_, axis=0), c='k')
        if f is not None:
            plt.plot(x[:, 0], f(x))
        plt.scatter(X[:, 0], Y)
        plt.title("HMC")
        plt.show()


if __name__ == '__main__':
    evalGPRSample()

    X, Y, x, f = data.make_data()

    evalHMC(X, Y, x, f)
    evalMLE(X, Y, x, f)
