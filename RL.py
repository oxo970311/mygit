import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

env = gym.make("CartPole-v1", render_mode="rgb_array")

obs, info = env.reset(seed=42)
print(obs)

img = env.render()
print(img.shape)


def plot_environment(env, figsize=(5, 4)):
    plt.figure(figsize=figsize)
    img = env.render()
    plt.imshow(img)
    plt.axis("off")
    return img


# plot_environment(env)
# plt.show()

print(env.action_space)

action = 1
obs, reward, done, truncated, info = env.step(action)
print(obs, reward, done, truncated, info)


def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1


totals = []
for episode in range(500):
    episode_reward = 0
    obs, info = env.reset(seed=episode)
    for step in range(200):
        action = basic_policy(obs)
        obs, reward, done, turncated, info = env.step(action)
        episode_reward += reward
        if done or turncated:
            break

    totals.append(episode_reward)

print(np.mean(totals), np.std(totals), min(totals), max(totals))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])


def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        left_proba = model(obs[np.newaxis])
        action = (tf.random.uniform([1, 1]) > left_proba)
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))

    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, turncated, info = env.step(int(action))
    return obs, reward, done, turncated, grads


def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        obs, info = env.reset()
        for step in range(n_max_steps):
            obs, reward, done, turncated, grads = play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done or turncated:
                break

        all_rewards.append(current_rewards)
        all_grads.append(current_grads)

    return all_rewards, all_grads


def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_factor
    return discounted


def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discount_rewards = [discount_rewards(rewards, discount_factor) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discount_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discount_rewards]


dr = discount_rewards([10, 0, -50], discount_factor=0.8)

dn = discount_and_normalize_rewards([[10, 0, -50], [10, 20]], discount_factor=0.8)

print(dr, dn)

n_iterations = 150
n_episodes_per_update = 10
n_max_steps = 200
discount_factor = 0.95

optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01)
loss_fn = tf.keras.losses.binary_crossentropy

# for iteration in range(n_iterations):
#     all_rewards, all_grads = play_multiple_episodes(env, n_episodes_per_update, n_max_steps, model, loss_fn)
#     all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_factor)
#     all_mean_grads = []
#     for var_index in range(len(model.trainable_variables)):
#         mean_grads = tf.reduce_mean([final_reward * all_grads[episode_index][step][var_index]
#             for episode_index, final_rewards in enumerate(all_final_rewards)
#                 for step, final_reward in enumerate(final_rewards)], axis=0)
#         all_mean_grads.append(mean_grads)
#     optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))

np.random.seed(42)

transition_probabilities = [  # shape=[s, s']
    [0.7, 0.2, 0.0, 0.1],  # s0에서 s0, s1, s2, s3로
    [0.0, 0.0, 0.9, 0.1],  # s1에서 s0, s1, s2, s3로
    [0.0, 1.0, 0.0, 0.0],  # s2에서 s0, s1, s2, s3로
    [0.0, 0.0, 0.0, 1.0]]  # s3에서 s0, s1, s2, s3로

n_max_steps = 1000  # 무한 루프 발생을 방지합니다.
terminal_states = [3]


def run_chain(start_state):
    current_state = start_state
    for step in range(n_max_steps):
        print(current_state, end=" ")
        if current_state in terminal_states:
            break
        current_state = np.random.choice(
            range(len(transition_probabilities)),
            p=transition_probabilities[current_state]
        )
    else:
        print("...", end="")

    print()


for idx in range(10):
    print(f"실행 #{idx + 1}: ", end="")
    run_chain(start_state=0)
