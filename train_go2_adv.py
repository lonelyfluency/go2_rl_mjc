from utils import *
from typing import List

#@title Barkour vb Quadruped Env

TIMESTEPS = 100_000
NUM_ENVS = 1024

def get_config():
  """Returns reward config for go2 quadruped environment."""

  def get_default_rewards_config():
    default_config = config_dict.ConfigDict(
        dict(
            # The coefficients for all reward terms used for training. All
            # physical quantities are in SI units, if no otherwise specified,
            # i.e. joint positions are in rad, positions are measured in meters,
            # torques in Nm, and time in seconds, and forces in Newtons.
            scales=config_dict.ConfigDict(
                dict(
                    # Tracking rewards are computed using exp(-delta^2/sigma)
                    # sigma can be a hyperparameters to tune.
                    # Track the base x-y velocity (no z-velocity tracking.)
                    tracking_lin_vel=1.5,
                    # Track the angular velocity along z-axis, i.e. yaw rate.
                    tracking_ang_vel=0.8,
                    # Below are regularization terms, we roughly divide the
                    # terms to base state regularizations, joint
                    # regularizations, and other behavior regularizations.
                    # Penalize the base velocity in z direction, L2 penalty.
                    lin_vel_z=-2.0,
                    # Penalize the base roll and pitch rate. L2 penalty.
                    ang_vel_xy=-0.05,
                    # Penalize non-zero roll and pitch angles. L2 penalty.
                    orientation=-5.0,
                    # L2 regularization of joint torques, |tau|^2.
                    torques=-0.0002,
                    # Penalize the change in the action and encourage smooth
                    # actions. L2 regularization |action - last_action|^2
                    action_rate=-0.01,
                    # Encourage long swing steps.  However, it does not
                    # encourage high clearances.
                    feet_air_time=0.2,
                    # Encourage no motion at zero command, L2 regularization
                    # |q - q_default|^2.
                    stand_still=-0.5,
                    # Early termination penalty.
                    termination=-1.0,
                    # Penalizing foot slipping on the ground.
                    foot_slip=-0.1,
                )
            ),
            # Tracking reward = exp(-error^2/sigma).
            tracking_sigma=0.25,
        )
    )
    return default_config

  default_config = config_dict.ConfigDict(
      dict(
          rewards=get_default_rewards_config(),
      )
  )

  return default_config


class Go2Env(PipelineEnv):
  """Environment for training the go2 quadruped joystick policy in MJX."""

  def __init__(
      self,
      obs_noise: float = 0.05,
      action_scale: float = 0.3,
      kick_vel: float = 0.05,
      **kwargs,
  ):
    path = epath.Path('resources/unitree_go2/scene_mjx.xml')
    sys = mjcf.load(path.as_posix())
    self._dt = 0.02  # this environment is 50 fps
    sys = sys.tree_replace({'opt.timestep': 0.004})
    # sys = sys.tree_replace({'opt.timestep': 0.004, 'dt': 0.004})

    # override menagerie params for smoother policy
    sys = sys.replace(
        dof_damping=sys.dof_damping.at[6:].set(0.5239),
        actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(35.0),
        actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-35.0),
    )

    n_frames = kwargs.pop('n_frames', int(self._dt / sys.opt.timestep))
    super().__init__(sys, backend='mjx', n_frames=n_frames)

    self.reward_config = get_config()
    # set custom from kwargs
    for k, v in kwargs.items():
      if k.endswith('_scale'):
        self.reward_config.rewards.scales[k[:-6]] = v

    self._torso_idx = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'base'
    )
    self._action_scale = action_scale
    self._obs_noise = obs_noise
    self._kick_vel = kick_vel
    self._init_q = jp.array(sys.mj_model.keyframe('home').qpos)
    self._default_pose = sys.mj_model.keyframe('home').qpos[7:]
    self.lowers = jp.array([-0.6, -1.4, -2.5] * 4)
    self.uppers = jp.array([0.6, 2.5, -0.85] * 4)
    feet_site = [
        'FL_foot',
        'RL_foot',
        'FR_foot',
        'RR_foot',
    ]
    feet_site_id = [
        mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
        for f in feet_site
    ]
    assert not any(id_ == -1 for id_ in feet_site_id), 'Site not found.'
    self._feet_site_id = np.array(feet_site_id)
    lower_leg_body = [
        'FL_calf',
        'RL_calf',
        'FR_calf',
        'RR_calf',
    ]
    lower_leg_body_id = [
        mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, l)
        for l in lower_leg_body
    ]
    assert not any(id_ == -1 for id_ in lower_leg_body_id), 'Body not found.'
    self._lower_leg_body_id = np.array(lower_leg_body_id)
    self._foot_radius = 0.0175
    self._nv = sys.nv

  def sample_command(self, rng: jax.Array) -> jax.Array:
    lin_vel_x = [-0.6, 1.5]  # min max [m/s]
    lin_vel_y = [-0.8, 0.8]  # min max [m/s]
    ang_vel_yaw = [-0.7, 0.7]  # min max [rad/s]

    _, key1, key2, key3 = jax.random.split(rng, 4)
    lin_vel_x = jax.random.uniform(
        key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
    )
    lin_vel_y = jax.random.uniform(
        key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
    )
    ang_vel_yaw = jax.random.uniform(
        key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
    )
    new_cmd = jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
    return new_cmd

  def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
    rng, key = jax.random.split(rng)

    pipeline_state = self.pipeline_init(self._init_q, jp.zeros(self._nv))

    state_info = {
        'rng': rng,
        'last_act': jp.zeros(12),
        'last_vel': jp.zeros(12),
        'command': self.sample_command(key),
        'last_contact': jp.zeros(4, dtype=bool),
        'feet_air_time': jp.zeros(4),
        'rewards': {k: 0.0 for k in self.reward_config.rewards.scales.keys()},
        'kick': jp.array([0.0, 0.0]),
        'step': 0,
    }

    obs_history = jp.zeros(15 * 31)  # store 15 steps of history
    obs = self._get_obs(pipeline_state, state_info, obs_history)
    reward, done = jp.zeros(2)
    metrics = {'total_dist': 0.0}
    for k in state_info['rewards']:
      metrics[k] = state_info['rewards'][k]
    state = State(pipeline_state, obs, reward, done, metrics, state_info)  # pytype: disable=wrong-arg-types
    return state

  def step(self, state: State, action_swing: jax.Array, action_stance: jax.Array) -> State:  # pytype: disable=signature-mismatch
    rng, cmd_rng, kick_noise_2 = jax.random.split(state.info['rng'], 3)

    # Kick logic (unchanged)
    push_interval = 10
    kick_theta = jax.random.uniform(kick_noise_2, maxval=2 * jp.pi)
    kick = jp.array([jp.cos(kick_theta), jp.sin(kick_theta)])
    kick *= jp.mod(state.info['step'], push_interval) == 0
    qvel = state.pipeline_state.qvel  # pytype: disable=attribute-error
    qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
    state = state.tree_replace({'pipeline_state.qvel': qvel})

    # Foot contact data
    foot_pos = pipeline_state.site_xpos[self._feet_site_id]  # pytype: disable=attribute-error
    foot_contact_z = foot_pos[:, 2] - self._foot_radius
    contact = foot_contact_z < 1e-3  # A mm or less off the floor (leg in contact)
    contact_filt_mm = contact | state.info['last_contact']
    contact_filt_cm = (foot_contact_z < 3e-2) | state.info['last_contact']
    first_contact = (state.info['feet_air_time'] > 0) * contact_filt_mm
    state.info['feet_air_time'] += self.dt

    # Leg state determination based on contact data (stance if in contact, otherwise swing)
    leg_states = ["stance" if contact[i] else "swing" for i in range(len(contact))]

    # Initialize a 12-dimensional action array
    combined_action = jp.zeros_like(action_swing)  # Action space is 12D (3 actions for each of 4 legs)

    # Assign actions from swing_agent or stance_agent based on the leg state
    for i, leg_state in enumerate(leg_states):
        if leg_state == "swing":
            combined_action = combined_action.at[i * 3:(i + 1) * 3].set(action_swing[i * 3:(i + 1) * 3])
        else:
            combined_action = combined_action.at[i * 3:(i + 1) * 3].set(action_stance[i * 3:(i + 1) * 3])

    # Physics step
    motor_targets = self._default_pose + combined_action * self._action_scale
    motor_targets = jp.clip(motor_targets, self.lowers, self.uppers)
    pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
    x, xd = pipeline_state.x, pipeline_state.xd

    # Observation data
    obs = self._get_obs(pipeline_state, state.info, state.obs)
    joint_angles = pipeline_state.q[7:]
    joint_vel = pipeline_state.qd[6:]

    # Check if the episode is done (falling or joint limits exceeded)
    up = jp.array([0.0, 0.0, 1.0])
    done = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
    done |= jp.any(joint_angles < self.lowers)
    done |= jp.any(joint_angles > self.uppers)
    done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18

    # Calculate rewards for swing and stance agents separately
    rewards_swing = {
        'lin_vel_z': self._reward_lin_vel_z(xd),
        'ang_vel_xy': self._reward_ang_vel_xy(xd),
        'orientation': self._reward_orientation(x),
        'feet_air_time': self._reward_feet_air_time(
            state.info['feet_air_time'],
            first_contact,
            state.info['command'],
        ),
    }

    rewards_stance = {
        'foot_slip': self._reward_foot_slip(pipeline_state, contact_filt_cm),
    }

    # Shared rewards for both agents
    shared_rewards = {
        'tracking_lin_vel': self._reward_tracking_lin_vel(state.info['command'], x, xd),
        'tracking_ang_vel': self._reward_tracking_ang_vel(state.info['command'], x, xd),
        'torques': self._reward_torques(pipeline_state.qfrc_actuator),  # pytype: disable=attribute-error
        'action_rate': self._reward_action_rate(combined_action, state.info['last_act']),
        'stand_still': self._reward_stand_still(state.info['command'], joint_angles),
        'termination': self._reward_termination(done, state.info['step']),
    }

    # Apply reward scaling and sum the rewards for both agents
    scaled_swing_rewards = {k: v * self.reward_config.rewards.scales.get(f"swing_{k}", 1.0) for k, v in rewards_swing.items()}
    scaled_stance_rewards = {k: v * self.reward_config.rewards.scales.get(f"stance_{k}", 1.0) for k, v in rewards_stance.items()}
    scaled_shared_rewards = {k: v * self.reward_config.rewards.scales.get(k, 1.0) for k, v in shared_rewards.items()}

    total_reward_swing = sum(scaled_swing_rewards.values())
    total_reward_stance = sum(scaled_stance_rewards.values())
    total_shared_reward = sum(scaled_shared_rewards.values())

    # Total reward for this step
    total_reward = (total_reward_swing + total_reward_stance + total_shared_reward) * self.dt
    total_reward = jp.clip(total_reward, 0.0, 10000.0)

    # State management
    state.info['kick'] = kick
    state.info['last_act'] = combined_action
    state.info['last_vel'] = joint_vel
    state.info['feet_air_time'] *= ~contact_filt_mm
    state.info['last_contact'] = contact
    state.info['rewards'] = {'swing': total_reward_swing, 'stance': total_reward_stance, 'shared': total_shared_reward}
    state.info['step'] += 1
    state.info['rng'] = rng

    # Sample new command if more than 500 timesteps achieved
    state.info['command'] = jp.where(
        state.info['step'] > 500,
        self.sample_command(cmd_rng),
        state.info['command'],
    )
    # Reset the step counter when done
    state.info['step'] = jp.where(
        done | (state.info['step'] > 500), 0, state.info['step']
    )

    # Log total displacement as a proxy metric
    state.metrics['total_dist'] = math.normalize(x.pos[self._torso_idx - 1])[1]
    state.metrics.update(state.info['rewards'])

    done = jp.float32(done)
    state = state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=total_reward, done=done
    )
    return state

  def _get_obs(
      self,
      pipeline_state: base.State,
      state_info: dict[str, Any],
      obs_history: jax.Array,
  ) -> jax.Array:
    inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
    local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

    obs = jp.concatenate([
        jp.array([local_rpyrate[2]]) * 0.25,                 # yaw rate
        math.rotate(jp.array([0, 0, -1]), inv_torso_rot),    # projected gravity
        state_info['command'] * jp.array([2.0, 2.0, 0.25]),  # command
        pipeline_state.q[7:] - self._default_pose,           # motor angles
        state_info['last_act'],                              # last action
    ])

    # clip, noise
    obs = jp.clip(obs, -100.0, 100.0) + self._obs_noise * jax.random.uniform(
        state_info['rng'], obs.shape, minval=-1, maxval=1
    )
    # stack observations through time
    obs = jp.roll(obs_history, obs.size).at[:obs.size].set(obs)

    return obs

  # ------------ reward functions----------------
  def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
    # Penalize z axis base linear velocity
    return jp.square(xd.vel[0, 2])

  def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
    # Penalize xy axes base angular velocity
    return jp.sum(jp.square(xd.ang[0, :2]))

  def _reward_orientation(self, x: Transform) -> jax.Array:
    # Penalize non flat base orientation
    up = jp.array([0.0, 0.0, 1.0])
    rot_up = math.rotate(up, x.rot[0])
    return jp.sum(jp.square(rot_up[:2]))

  def _reward_torques(self, torques: jax.Array) -> jax.Array:
    # Penalize torques
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _reward_action_rate(
      self, act: jax.Array, last_act: jax.Array
  ) -> jax.Array:
    # Penalize changes in actions
    return jp.sum(jp.square(act - last_act))

  def _reward_tracking_lin_vel(
      self, commands: jax.Array, x: Transform, xd: Motion
  ) -> jax.Array:
    # Tracking of linear velocity commands (xy axes)
    local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    lin_vel_reward = jp.exp(
        -lin_vel_error / self.reward_config.rewards.tracking_sigma
    )
    return lin_vel_reward

  def _reward_tracking_ang_vel(
      self, commands: jax.Array, x: Transform, xd: Motion
  ) -> jax.Array:
    # Tracking of angular velocity commands (yaw)
    base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
    ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
    return jp.exp(-ang_vel_error / self.reward_config.rewards.tracking_sigma)

  def _reward_feet_air_time(
      self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
  ) -> jax.Array:
    # Reward air time.
    rew_air_time = jp.sum((air_time - 0.1) * first_contact)
    rew_air_time *= (
        math.normalize(commands[:2])[1] > 0.05
    )  # no reward for zero command
    return rew_air_time

  def _reward_stand_still(
      self,
      commands: jax.Array,
      joint_angles: jax.Array,
  ) -> jax.Array:
    # Penalize motion at zero commands
    return jp.sum(jp.abs(joint_angles - self._default_pose)) * (
        math.normalize(commands[:2])[1] < 0.1
    )

  def _reward_foot_slip(
      self, pipeline_state: base.State, contact_filt: jax.Array
  ) -> jax.Array:
    # get velocities at feet which are offset from lower legs
    # pytype: disable=attribute-error
    pos = pipeline_state.site_xpos[self._feet_site_id]  # feet position
    feet_offset = pos - pipeline_state.xpos[self._lower_leg_body_id]
    # pytype: enable=attribute-error
    offset = base.Transform.create(pos=feet_offset)
    foot_indices = self._lower_leg_body_id - 1  # we got rid of the world body
    foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

    # Penalize large feet velocity for feet that are in contact with the ground.
    return jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))

  def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
    return done & (step < 500)

  def render(
      self, trajectory: List[base.State], camera: str | None = None
  ) -> Sequence[np.ndarray]:
    camera = camera or 'track'
    return super().render(trajectory, camera=camera)

envs.register_environment('go2', Go2Env)
env_name = 'go2'
env = envs.get_environment(env_name)

# Create the two-agent (swing and stance) network factory
def make_ippo_networks_factory(observation_size, action_size):
    """Creates network factories for both swing and stance agents."""
    
    # Define swing and stance agent network factories to accept positional args directly
    def swing_agent_network_factory(observation_size, action_size, **kwargs):
        return ppo_networks.make_ppo_networks(
            observation_size=observation_size,
            action_size=action_size,
            policy_hidden_layer_sizes=(128, 128, 128, 128),
            **kwargs
        )
    
    def stance_agent_network_factory(observation_size, action_size, **kwargs):
        return ppo_networks.make_ppo_networks(
            observation_size=observation_size,
            action_size=action_size,
            policy_hidden_layer_sizes=(128, 128, 128, 128),
            **kwargs
        )
    
    # Return both factories, each accepting observation and action sizes as args
    return {
        "swing_agent": swing_agent_network_factory,
        "stance_agent": stance_agent_network_factory
    }



# Train function to support the two agents
def train_ippo_agents(env, eval_env, num_timesteps=100_000_000, **kwargs):
    """Trains two agents (swing and stance) using IPPO and updates their networks independently."""
    
    # Get observation size and action size from the environment
    dummy_reset = env.reset(jax.random.PRNGKey(0))
    observation_size = dummy_reset.obs.shape[0]
    action_size = env.action_size

    # Create agents using the factory
    agents = make_ippo_networks_factory(observation_size, action_size)

    # Setup progress bars
    progress_bar_swing = tqdm(total=num_timesteps, desc='Swing Agent Training Progress')
    progress_bar_stance = tqdm(total=num_timesteps, desc='Stance Agent Training Progress')

    def progress_swing(num_steps, metrics):
        """Logs and visualizes progress for swing agent."""
        times.append(datetime.now())
        x_data_swing.append(num_steps)
        y_data_swing.append(metrics['eval/episode_reward'])
        ydataerr_swing.append(metrics['eval/episode_reward_std'])

        plt.figure(1)
        plt.xlim([0, num_timesteps * 1.25])
        plt.ylim([min_y, max_y])

        plt.xlabel('# environment steps')
        plt.ylabel('Swing Agent Reward per Episode')
        plt.title(f'Swing Agent Reward: {y_data_swing[-1]:.3f}')

        plt.errorbar(x_data_swing, y_data_swing, yerr=ydataerr_swing)
        plt.savefig("training_swing_agent_go2.png")
        progress_bar_swing.update(num_steps - progress_bar_swing.n)

    def progress_stance(num_steps, metrics):
        """Logs and visualizes progress for stance agent."""
        times.append(datetime.now())
        x_data_stance.append(num_steps)
        y_data_stance.append(metrics['eval/episode_reward'])
        ydataerr_stance.append(metrics['eval/episode_reward_std'])

        plt.figure(2)
        plt.xlim([0, num_timesteps * 1.25])
        plt.ylim([min_y, max_y])

        plt.xlabel('# environment steps')
        plt.ylabel('Stance Agent Reward per Episode')
        plt.title(f'Stance Agent Reward: {y_data_stance[-1]:.3f}')

        plt.errorbar(x_data_stance, y_data_stance, yerr=ydataerr_stance)
        plt.savefig("training_stance_agent_go2.png")
        progress_bar_stance.update(num_steps - progress_bar_stance.n)

    # Train swing and stance agents independently
    swing_agent_params, _ = ppo.train(
        environment=env,
        num_timesteps=num_timesteps,
        network_factory=agents["swing_agent"],  # Use the swing agent factory directly
        progress_fn=progress_swing,
        eval_env=eval_env,
        **kwargs
    )

    stance_agent_params, _ = ppo.train(
        environment=env,
        num_timesteps=num_timesteps,
        network_factory=agents["stance_agent"],  # Use the stance agent factory directly
        progress_fn=progress_stance,
        eval_env=eval_env,
        **kwargs
    )

    # Close the progress bars
    progress_bar_swing.close()
    progress_bar_stance.close()

    return swing_agent_params, stance_agent_params


# Initialize logging and visualization
x_data_swing = []
x_data_stance = []
y_data_swing = []
y_data_stance = []
ydataerr_swing = []
ydataerr_stance = []
times = [datetime.now()]
max_y, min_y = 40, 0

# Initialize tqdm progress bar
progress_bar_swing = tqdm(total=TIMESTEPS, desc='Swing Agent Training Progress')
progress_bar_stance = tqdm(total=TIMESTEPS, desc='Stance Agent Training Progress')



# Reset environments and initialize inference function
env = envs.get_environment(env_name)
eval_env = envs.get_environment(env_name)

# Train both swing and stance agents
swing_agent_params, stance_agent_params = train_ippo_agents(
    env=env,
    eval_env=eval_env,
    num_timesteps=TIMESTEPS,
    num_evals=10,
    reward_scaling=1,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.97,
    learning_rate=3.0e-4,
    entropy_cost=1e-2,
    num_envs=NUM_ENVS,
    batch_size=256,
    randomization_fn=domain_randomize,
    seed=0
)

print(f'Time to JIT: {times[1] - times[0]}')
print(f'Time to train: {times[-1] - times[1]}')

# Save and reload parameters for both agents
model_path_swing = './tmp/mjx_brax_go2_swing_policy'
model.save_params(model_path_swing, swing_agent_params)
swing_agent_params = model.load_params(model_path_swing)

model_path_stance = './tmp/mjx_brax_go2_stance_policy'
model.save_params(model_path_stance, stance_agent_params)
stance_agent_params = model.load_params(model_path_stance)

# Get the inference functions for the two agents
swing_inference_fn = jax.jit(ppo_networks.make_inference_fn(swing_agent_params))
stance_inference_fn = jax.jit(ppo_networks.make_inference_fn(stance_agent_params))

# Environment reset and step functions
eval_env = envs.get_environment(env_name)
jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

# Commands for Barkour Env
x_vel = 1.0  # Example command input (x velocity)
y_vel = 0.0  # Example command input (y velocity)
ang_vel = -0.1  # Example angular velocity input

the_command = jp.array([x_vel, y_vel, ang_vel])

# Initialize the environment state
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
state.info['command'] = the_command
rollout = [state.pipeline_state]

# Rollout trajectory and render the video
n_steps = 500
render_every = 2

# Run inference and simulation loop
for i in range(n_steps):
    # Split RNG for both agents to ensure each receives a unique key
    act_rng_swing, act_rng_stance, rng = jax.random.split(rng, 3)
    
    # Generate actions from both swing and stance agents
    action_swing, _ = swing_inference_fn(state.obs, act_rng_swing)
    action_stance, _ = stance_inference_fn(state.obs, act_rng_stance)
    
    # Apply actions in the environment using the updated step function for swing and stance actions
    state = jit_step(state, action_swing, action_stance)
    
    # Collect rollout for visualization
    rollout.append(state.pipeline_state)

# Save the video of the rollout
save_video(
    eval_env.render(rollout[::render_every], camera='track'),
    fps=1.0 / eval_env.dt / render_every
)