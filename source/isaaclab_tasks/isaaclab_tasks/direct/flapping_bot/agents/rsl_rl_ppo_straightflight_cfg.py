"""RSL-RL PPO configuration for FlappingBot straight-flight tasks."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class FlappingBotStraightFlightPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Rollout and training schedule
    num_steps_per_env = 48
    max_iterations = 4000
    save_interval = 200
    experiment_name = "flapping_bot_straight_flight"

    # Policy network
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )

    # PPO algorithm
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
