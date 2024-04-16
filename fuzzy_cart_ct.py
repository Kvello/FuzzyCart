import gymnasium as gym
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np


PLOT = True
DEBUG = False

# Initialize the mountain car environment
env = gym.make('MountainCarContinuous-v0', render_mode="human")

# Define the input variables: position and velocity
position_universe = np.arange(-0.9, 1, 0.1)
velocity_universe = np.arange(-0.07, 0.08, 0.01)

# Define the output variable: action (left, neutral, right)
action_universe = np.arange(-1, 1.1, 0.1)


# Define the fuzzy sets for the input and output variables
position_ne = fuzz.trimf(position_universe, [-0.9, -0.9, 0])
position_ze = fuzz.trimf(position_universe, [-0.6, 0, 0.6])
position_po = fuzz.trimf(position_universe, [0, 0.9, 0.9])

velocity_ne = fuzz.trimf(velocity_universe, [-0.07, -0.07, 0])
velocity_ze = fuzz.trimf(velocity_universe, [-0.035, 0, 0.035])
velocity_po = fuzz.trimf(velocity_universe, [0, 0.07, 0.07])

output_ne = fuzz.trimf(action_universe, [-1, -1, -0.3])
output_ze = fuzz.trimf(action_universe, [-0.6, 0, 0.6])
output_po = fuzz.trimf(action_universe, [0.3, 1, 1])

# Plot the fuzzy sets if PLOT is True
if PLOT:
    import matplotlib.pyplot as plt
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

    ax0.plot(position_universe, position_ne, 'b',
             linewidth=1.5, label='Negative')
    ax0.plot(position_universe, position_ze, 'g', linewidth=1.5, label='Zero')
    ax0.plot(position_universe, position_po, 'r',
             linewidth=1.5, label='Positive')
    ax0.set_title('Position')
    ax0.legend()

    ax1.plot(velocity_universe, velocity_ne, 'b',
             linewidth=1.5, label='Negative')
    ax1.plot(velocity_universe, velocity_ze, 'g', linewidth=1.5, label='Zero')
    ax1.plot(velocity_universe, velocity_po, 'r',
             linewidth=1.5, label='Positive')
    ax1.set_title('Velocity')
    ax1.legend()

    ax2.plot(action_universe, output_ne, 'b', linewidth=1.5, label='Negative')
    ax2.plot(action_universe, output_ze, 'g', linewidth=1.5, label='Zero')
    ax2.plot(action_universe, output_po, 'r', linewidth=1.5, label='Positive')
    ax2.set_title('Throttle')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# Define the controller function


def fuzzy_controller(observation):
    position_level_ne = fuzz.interp_membership(
        position_universe, position_ne, observation[0] + 0.3)
    position_level_ze = fuzz.interp_membership(
        position_universe, position_ze, observation[0] + 0.3)
    position_level_po = fuzz.interp_membership(
        position_universe, position_po, observation[0] + 0.3)
    velocity_level_ne = fuzz.interp_membership(
        velocity_universe, velocity_ne, observation[1])
    velocity_level_ze = fuzz.interp_membership(
        velocity_universe, velocity_ze, observation[1])
    velocity_level_po = fuzz.interp_membership(
        velocity_universe, velocity_po, observation[1])

    rule0_activation = np.fmax(velocity_level_ne, np.fmin(
        position_level_po, velocity_level_ze))
    rule1_activation = np.fmax(np.fmax(velocity_level_po, np.fmin(
        position_level_ne, velocity_level_ze)), np.fmin(velocity_level_ze, position_level_ze))
    rule2_activation = np.fmin(velocity_level_po, position_level_po)

    if DEBUG:
        print("position level ne: ", position_level_ne)
        print("position level ze: ", position_level_ze)
        print("position level po: ", position_level_po)
        print("velocity level ne: ", velocity_level_ne)
        print("velocity level ze: ", velocity_level_ze)
        print("velocity level po: ", velocity_level_po)
        print("Rule0: ", rule0_activation)
        print("Rule1: ", rule1_activation)
        print("Rule2: ", rule2_activation)

    ne_activation = np.fmin(rule0_activation, output_ne)
    po_activation = np.fmin(rule1_activation, output_po)
    ze_activation = np.fmin(rule2_activation, output_ze)

    aggregated = np.fmax(ne_activation, np.fmax(po_activation, ze_activation))
    if np.sum(aggregated) == 0:
        action = 0
    else:
        action = fuzz.defuzz(action_universe, aggregated, 'centroid')
    action = np.array([action])

    return action


# Run the simulation
observation, info = env.reset()
done = False
while not done:
    env.render()
    # action = fuzzy_controller(observation)
    action = fuzzy_controller(observation)
    print("Observation: ", observation)
    observation, reward, done, _, info = env.step(action)
env.close()
