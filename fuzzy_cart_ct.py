import gymnasium as gym
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import tqdm

PLOT = False
DEBUG = True
CONTROLLER = "THREE_RULE"
controllers = ["THREE_RULE", "THREE_RULE_REDUCED",
               "TWO_RULE", "TWO_RULE_REDUCED"]

N = 10000  # Number of episodes to run for each controller
# Set to True to run the benchmark, set to false to render one episode of the controller
BENCHMARK = False

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
velocity_ze = fuzz.trimf(velocity_universe, [-0.03, 0, 0.03])
velocity_po = fuzz.trimf(velocity_universe, [0, 0.07, 0.07])

output_ne = fuzz.trimf(action_universe, [-1, -1, -0.3])
output_ze = fuzz.trimf(action_universe, [-0.6, 0, 0.6])
output_po = fuzz.trimf(action_universe, [0.3, 1, 1])


def save_results(results, filename):
    with open(filename, "w") as f:
        f.write("Results for the fuzzy controllers using N="+str(N)+"episodes\n")
        for controller, result in results.items():
            f.write(controller + "\n")
            f.write("Average reward: " + str(result["average_reward"]) + "\n")
            f.write("Std reward: " + str(result["std_reward"]) + "\n")
            f.write("95\% confidence interval: +\/-" +
                    str(1.96*result["std_reward"]/np.sqrt(N)) + '\n')
            f.write("Min reward: " + str(result["min_reward"]) + "\n")
            f.write("Max reward: " + str(result["max_reward"]) + "\n")
            f.write("\n")


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

# Define the controller functions


def fuzzy_three_rule_controller(pos_level_ne, pos_level_ze, pos_level_po, vel_level_ne, vel_level_ze, vel_level_po):

    rule0_activation = np.fmax(vel_level_ne, np.fmin(
        pos_level_po, vel_level_ze))
    rule1_activation = np.fmax(np.fmax(vel_level_po, np.fmin(
        pos_level_ne, vel_level_ze)), np.fmin(vel_level_ze, pos_level_ze))
    rule2_activation = np.fmin(vel_level_po, pos_level_po)

    if DEBUG:
        print("position level ne: ", pos_level_ne)
        print("position level ze: ", pos_level_ze)
        print("position level po: ", pos_level_po)
        print("velocity level ne: ", vel_level_ne)
        print("velocity level ze: ", vel_level_ze)
        print("velocity level po: ", vel_level_po)
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


def fuzzy_three_rule_reduced_controller(pos_level_ne, pos_level_ze, pos_level_po, vel_level_ne, vel_level_ze, vel_level_po):

    rule0_activation = vel_level_ne
    rule1_activation = np.fmax(
        vel_level_po, np.fmin(vel_level_ze, pos_level_ze))
    rule2_activation = np.fmin(vel_level_po, pos_level_po)

    if DEBUG:
        print("position level ne: ", pos_level_ne)
        print("position level ze: ", pos_level_ze)
        print("position level po: ", pos_level_po)
        print("velocity level ne: ", vel_level_ne)
        print("velocity level ze: ", vel_level_ze)
        print("velocity level po: ", vel_level_po)
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


def fuzzy_two_rule_controller(pos_level_ne, pos_level_ze, pos_level_po, vel_level_ne, vel_level_ze, vel_level_po):

    rule0_activation = np.fmax(vel_level_ne, np.fmin(
        pos_level_po, vel_level_ze))
    rule1_activation = np.fmax(np.fmax(vel_level_po, np.fmin(
        pos_level_ne, vel_level_ze)), np.fmin(vel_level_ze, pos_level_ze))
    if DEBUG:
        print("position level ne: ", pos_level_ne)
        print("position level ze: ", pos_level_ze)
        print("position level po: ", pos_level_po)
        print("velocity level ne: ", vel_level_ne)
        print("velocity level ze: ", vel_level_ze)
        print("velocity level po: ", vel_level_po)
        print("Rule0: ", rule0_activation)
        print("Rule1: ", rule1_activation)

    ne_activation = np.fmin(rule0_activation, output_ne)
    po_activation = np.fmin(rule1_activation, output_po)

    aggregated = np.fmax(ne_activation, po_activation)
    if np.sum(aggregated) == 0:
        action = 0
    else:
        action = fuzz.defuzz(action_universe, aggregated, 'centroid')
    action = np.array([action])

    return action


def fuzzy_two_rule_reduced_controller(pos_level_ne, pos_level_ze, pos_level_po, vel_level_ne, vel_level_ze, vel_level_po):
    rule0_activation = vel_level_ne
    rule1_activation = np.fmax(
        vel_level_po, np.fmin(pos_level_ne, vel_level_ze))

    if DEBUG:
        print("position level ne: ", pos_level_ne)
        print("position level ze: ", pos_level_ze)
        print("position level po: ", pos_level_po)
        print("velocity level ne: ", vel_level_ne)
        print("velocity level ze: ", vel_level_ze)
        print("velocity level po: ", vel_level_po)
        print("Rule0: ", rule0_activation)
        print("Rule1: ", rule1_activation)

    ne_activation = np.fmin(rule0_activation, output_ne)
    po_activation = np.fmin(rule1_activation, output_po)

    aggregated = np.fmax(ne_activation, po_activation)
    if np.sum(aggregated) == 0:
        action = 0
    else:
        action = fuzz.defuzz(action_universe, aggregated, 'centroid')
    action = np.array([action])

    return action


def get_fuzzy_action(observation):
    if CONTROLLER == "PPO":
        return np.array([0])
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

    if CONTROLLER == "THREE_RULE":
        action = fuzzy_three_rule_controller(position_level_ne, position_level_ze, position_level_po,
                                             velocity_level_ne, velocity_level_ze, velocity_level_po)
    elif CONTROLLER == "THREE_RULE_REDUCED":
        action = fuzzy_three_rule_reduced_controller(position_level_ne, position_level_ze, position_level_po,
                                                     velocity_level_ne, velocity_level_ze, velocity_level_po)
    elif CONTROLLER == "TWO_RULE":
        action = fuzzy_two_rule_controller(position_level_ne, position_level_ze, position_level_po,
                                           velocity_level_ne, velocity_level_ze, velocity_level_po)
    elif CONTROLLER == "TWO_RULE_REDUCED":
        action = fuzzy_two_rule_reduced_controller(position_level_ne, position_level_ze, position_level_po,
                                                   velocity_level_ne, velocity_level_ze, velocity_level_po)

    return action


def main():
    if BENCHMARK:
        results = {}
        for controller in controllers:
            global CONTROLLER
            CONTROLLER = controller
            scores = np.zeros(N)
            print("Running controller: ", controller)
            for i in tqdm.tqdm(range(N)):
                observation, info = env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action = get_fuzzy_action(observation)
                    observation, reward, done, _, info = env.step(action)
                    episode_reward += reward
                scores[i] = episode_reward
            print("Controller: ", controller)
            print("Average reward: ", np.mean(scores))
            results[controller] = {
                "average_reward": np.mean(scores),
                "std_reward": np.std(scores),
                "min_reward": np.min(scores),
                "max_reward": np.max(scores)
            }
        save_results(results, "fuzzy_cart_ct_results.txt")
    else:
        observation, info = env.reset()
        done = False
        score = 0
        while not done:
            env.render()
            # action = fuzzy_controller(observation)
            action = get_fuzzy_action(observation)
            if DEBUG:
                print("Action: ", action)
                print("Observation: ", observation)
            observation, reward, done, _, info = env.step(action)
            print("Reward: ", reward)
            score += reward
        print("Score: ", score)
        env.close()


if __name__ == "__main__":
    main()
