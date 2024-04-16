import gymnasium as gym
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np

# Initialize the mountain car environment
env = gym.make('MountainCarContinuous-v0', render_mode="human")

# Define the input variables: position and velocity
position_universe = np.arange(-0.9, 1, 0.1)
velocity_universe = np.arange(-0.07, 0.08, 0.01)

# Define the output variable: action (left, neutral, right)
action_universe = np.arange(-1, 1.1, 0.1)

# Generate fuzzy membership functions for the input and output variables
# position = ctrl.Antecedent(position, 'position')
# delta = ctrl.Antecedent(position, 'delta')
# output = ctrl.Consequent(action, 'output')

# Visualize the membership functions (optional)
# fuzz.visualization.fuzzy_mf.plot_mfs(position, position_mf, velocity, velocity_mf, action, action_mf)
# names = ['ne', 'ze', 'po']
# position.automf(names=names)
# delta.automf(names=names)
# output.automf(names=names)

# Define the fuzzy sets for the input and output variables
position_ne = fuzz.trimf(position_universe, [-0.9, -0.9, 0])
position_ze = fuzz.trimf(position_universe, [-0.6, 0, 0.6])
position_po = fuzz.trimf(position_universe, [0, 0.9, 0.9])

delta_ne = fuzz.trimf(velocity_universe, [-0.07, -0.07, 0])
delta_ze = fuzz.trimf(velocity_universe, [-0.035, 0, 0.035])
delta_po = fuzz.trimf(velocity_universe, [0, 0.07, 0.07])

output_ne = fuzz.trimf(action_universe, [-1, -1, -0.3])
output_ze = fuzz.trimf(action_universe, [-0.6, 0, 0.6])
output_po = fuzz.trimf(action_universe, [0.3, 1, 1])

# rule0 = ctrl.Rule(antecedent=(delta['ne']|
#                               delta['ze']&position['po']),
#                     consequent=output['ne'], label='rule ne')
# rule1 = ctrl.Rule(antecedent=(delta['po']|
#                               delta['ze']&position['ne']),
#                     consequent=output['po'], label='rule po')
# rule2 = ctrl.Rule(antecedent=(delta['po']&position['po']),
#                     consequent=output['ze'], label='rule ze')

# controller = ctrl.ControlSystem([rule0, rule1, rule2])
# sim = ctrl.ControlSystemSimulation(controller)
# Define the controller function
def fuzzy_controller(observation):
    position_level_ne = fuzz.interp_membership(position_universe, position_ne, observation[0] + 0.3)
    position_level_ze = fuzz.interp_membership(position_universe, position_ze, observation[0] + 0.3)
    position_level_po = fuzz.interp_membership(position_universe, position_po, observation[0] + 0.3)
    delta_level_ne = fuzz.interp_membership(velocity_universe, delta_ne, observation[1])
    delta_level_ze = fuzz.interp_membership(velocity_universe, delta_ze, observation[1])
    delta_level_po = fuzz.interp_membership(velocity_universe, delta_po, observation[1])

    rule0_activation = np.fmax(delta_level_ne, np.fmin(position_level_po, delta_level_ze))
    rule1_activation = np.fmax(np.fmax(delta_level_po, np.fmin(position_level_ne, delta_level_ze)),np.fmin(delta_level_ze, position_level_ze))
    rule2_activation = np.fmin(delta_level_po, position_level_po)

    print("position level ne: ", position_level_ne)
    print("position level ze: ", position_level_ze)
    print("position level po: ", position_level_po)
    print("delta level ne: ", delta_level_ne)
    print("delta level ze: ", delta_level_ze)
    print("delta level po: ", delta_level_po)
    print("Rule0: ", rule0_activation)
    print("Rule1: ", rule1_activation)
    print("Rule2: ", rule2_activation)

    ne_activation = np.fmin(rule0_activation, output_ne)
    po_activation = np.fmin(rule1_activation, output_po)
    ze_activation = np.fmin(rule2_activation, output_ze)

    aggregated = np.fmax(ne_activation,np.fmax(po_activation, ze_activation))
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
    #action = fuzzy_controller(observation)
    action = fuzzy_controller(observation)
    print("Observation: ", observation)
    observation, reward, done, _, info = env.step(action)
env.close()