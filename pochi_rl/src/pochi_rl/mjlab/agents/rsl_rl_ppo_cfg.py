"""RSL-RL PPO configurations for the Pochi tasks."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def pochi_flat_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="pochi_velocity_flat",
    logger="tensorboard",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=1500,
  )


def pochi_standup_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO for the stand-up task.

  Same network as the velocity runner.  The differences are all about the task
  being a single manoeuvre rather than an endless gait.  Episodes are long
  relative to the rollout, so the horizon is stretched to 48 steps, and gamma is
  raised because getting up is worth a lot of reward that only arrives seconds
  after the actions that earned it.

  Entropy is twice the velocity runner's.  The task has one narrow solution at
  the end of a long sequence, so exploration has to outlive a gait's: at 0.005
  the action standard deviation collapses to 0.08 by iteration 300 and the run
  then grinds at 0.1 reward per hundred iterations.  0.02 is too far the other
  way -- the deviation never falls below 0.85 and the policy does not converge
  at all.  What actually made the task learnable was giving the reward a joint
  trajectory to track rather than only a height (see StandUpReferenceSpec);
  entropy is a second-order knob next to that.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.995,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="pochi_standup_flat",
    logger="tensorboard",
    save_interval=50,
    num_steps_per_env=48,
    max_iterations=1500,
  )
