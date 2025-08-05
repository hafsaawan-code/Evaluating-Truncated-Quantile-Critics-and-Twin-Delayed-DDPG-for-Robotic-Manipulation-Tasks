import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplyBuffer
from networks import ActorNetwork, QuantileCriticNetwork

def quantile_huber_loss(pred_quantiles, target_quantiles, taus, kappa=1.0):
    u = target_quantiles.unsqueeze(1) - pred_quantiles.unsqueeze(2)  # [batch, n_quantiles, n_quantiles]
    huber = T.where(u.abs() <= kappa, 0.5 * u.pow(2), kappa * (u.abs() - 0.5 * kappa))
    loss = T.abs(taus.unsqueeze(0) - (u.detach() < 0).float()) * huber / kappa
    return loss.sum(dim=2).mean(dim=1).mean()

class Agent:
    def __init__(
        self, actor_learning_rate, critic_learning_rate, input_dims, tau, env,
        gamma=0.99, update_actor_interval=2, warmup=1000, n_actions=2, max_size=1000000,
        layer1_size=256, layer2_size=128,total_episodes = 500, batch_size=100, noise=0.1, n_quantiles=25,
    ):
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplyBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval
        self.n_quantiles = n_quantiles

        self.total_episodes = total_episodes   # Set externally in training loop if needed
        self.noise_initial = noise

        # Create the networks
        self.actor = ActorNetwork(
            input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size,
            n_actions=n_actions, name='actor', learning_rate=actor_learning_rate
        )
        self.critic_1 = QuantileCriticNetwork(
            input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size,
            n_actions=n_actions, n_quantiles=self.n_quantiles, name='critic_1', learning_rate=critic_learning_rate
        )
        self.critic_2 = QuantileCriticNetwork(
            input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size,
            n_actions=n_actions, n_quantiles=self.n_quantiles, name='critic_2', learning_rate=critic_learning_rate
        )

        # Create the target networks
        self.target_actor = ActorNetwork(
            input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size,
            n_actions=n_actions, name='target_actor', learning_rate=actor_learning_rate
        )
        self.target_critic_1 = QuantileCriticNetwork(
            input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size,
            n_actions=n_actions, n_quantiles=self.n_quantiles, name='target_critic_1', learning_rate=critic_learning_rate
        )
        self.target_critic_2 = QuantileCriticNetwork(
            input_dims=input_dims, fc1_dims=layer1_size, fc2_dims=layer2_size,
            n_actions=n_actions, n_quantiles=self.n_quantiles, name='target_critic_2', learning_rate=critic_learning_rate
        )

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, validation=False, episode=0):
        if self.time_step < self.warmup and not validation:
            mu = T.tensor(np.random.normal(scale=self.noise_initial, size=(self.n_actions,))).to(self.actor.device)
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)

        # Anneal noise
        noise_std = self.noise_initial * (1 - min(1.0, episode / self.total_episodes))
        mu_prime = mu + T.tensor(np.random.normal(scale=noise_std), dtype=T.float).to(self.actor.device)
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def learn(self):
        if self.memory.mem_ctr < self.batch_size * 10:
            return
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
        reward = T.tensor(reward, dtype=T.float).unsqueeze(1).to(self.critic_1.device)
        done = T.tensor(done, dtype=T.float).unsqueeze(1).to(self.critic_1.device)
        next_state = T.tensor(next_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        taus = T.linspace(0.5 / self.n_quantiles, 1 - 0.5 / self.n_quantiles, self.n_quantiles, device=self.critic_1.device)

        # Target actions with noise (policy smoothing)
        target_actions = self.target_actor.forward(next_state)
        noise = (T.randn_like(target_actions) * 0.2).clamp(-0.5, 0.5)
        target_actions = (target_actions + noise).clamp(self.min_action[0], self.max_action[0])

        # Get target quantiles from both target critics
        next_q1 = self.target_critic_1(next_state, target_actions)
        next_q2 = self.target_critic_2(next_state, target_actions)
        next_q = T.min(next_q1, next_q2)  # elementwise min
        next_q = next_q * (1 - done)

        target_quantiles = reward + self.gamma * next_q
        target_quantiles = target_quantiles.detach()

        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        critic_1_loss = quantile_huber_loss(current_q1, target_quantiles, taus)
        critic_2_loss = quantile_huber_loss(current_q2, target_quantiles, taus)
        critic_loss = critic_1_loss + critic_2_loss

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        # Actor update: maximize mean of quantiles
        self.actor.optimizer.zero_grad()
        actions = self.actor(state)
        q1 = self.critic_1(state, actions)
        actor_loss = -q1.mean()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        try:
            self.actor.load_checkpoint()
            self.target_actor.load_checkpoint()
            self.critic_1.load_checkpoint()
            self.critic_2.load_checkpoint()
            self.target_critic_1.load_checkpoint()
            self.target_critic_2.load_checkpoint()
            print("successfully loaded the models")
        except:
            print("Failed to load the models. Starting from scratch")