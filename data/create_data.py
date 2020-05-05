import numpy as np

def sim_rollout(sim, policy, n_steps, dt, init_state):
    """
    :param sim: the simulator
    :param policy: policy that generates rollout
    :param n_steps: number of time steps to run
    :param dt: simulation step size
    :param init_state: initial state

    :return: times:   a numpy array of size [n_steps + 1]
             states:  a numpy array of size [n_steps + 1 x 4]
             actions: a numpy array of size [n_steps]
                        actions[i] is applied to states[i] to generate states[i+1]
    """
    states = []
    state = init_state
    actions = []
    for i in range(n_steps):
        states.append(state)
        action = policy.predict(state)
        actions.append(action)
        state = sim.step(state, [action], noisy=True)

    states.append(state)
    times = np.arange(n_steps + 1) * dt
    return times, np.array(states), np.array(actions)

def augmented_state(state, action):
    """
    :param state: cartpole state
    :param action: action applied to state
    :return: an augmented state for training GP dynamics
    """
    dtheta, dx, theta, x = state
    return x, dx, dtheta, np.sin(theta), np.cos(theta), action

def make_training_data(state_traj, action_traj, delta_state_traj):
    """
    A helper function to generate training data.
    """
    x = np.array([augmented_state(state, action) for state, action in zip(state_traj, action_traj)])
    y = delta_state_traj
    return x, y

# Global variables
DELTA_T = 0.05
NUM_DATAPOINTS_PER_EPOCH = 50
rng = np.random.RandomState(12345)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    from cartpole_sim import CartpoleSim
    from policy import SwingUpAndBalancePolicy, RandomPolicy
    from visualization import Visualizer
    import cv2

    vis = Visualizer(cartpole_length=1.5, x_lim=(0.0, DELTA_T * NUM_DATAPOINTS_PER_EPOCH))
    swingup_policy = SwingUpAndBalancePolicy('policy.npz')
    random_policy = RandomPolicy(seed=12831)
    sim = CartpoleSim(dt=DELTA_T)

    IMAGE_DATASET = True

    if not IMAGE_DATASET:
        NUM_DATAPOINTS = 11000
        '''
        we want numerical value data 
        '''
        # Initial training data used to train GP for the first epoch
        init_state = np.array([0.01, 0.01, np.pi * 0.5, 0.1]) * rng.randn(4)
        ts, state_traj, action_traj = sim_rollout(sim, random_policy, NUM_DATAPOINTS, DELTA_T, init_state)
        delta_state_traj = state_traj[1:] - state_traj[:-1]
        train_x, train_y = make_training_data(state_traj[:-1], action_traj, delta_state_traj)
        '''
        TODO: write code to save numerical data (train_x, train_y)
        '''
        data = np.concatenate((train_x, train_y), axis=1)
        np.savetxt('vanilla_dataset/data.csv', data, delimiter=',')
        # method to load data
        savedData = np.loadtxt('vanilla_dataset/data.csv', delimiter=',')

    if IMAGE_DATASET:
        import csv 
        import os
        NUM_TRAINING_EPOCHS = 220 # multiply by 50 to get num. data points
        PATH = "image_dataset/"
        '''
        we want image data
        '''
        with open(PATH + "data.csv", mode='w', newline='') as csv_file:    
            csv_writer = csv.writer(csv_file, delimiter=',')
            for epoch in range(NUM_TRAINING_EPOCHS):
                vis.clear()

                # Use learned policy every 4th epoch
                if (epoch + 1) % 4 == 0:
                    policy = swingup_policy
                    init_state = np.array([0.01, 0.01, 0.05, 0.05]) * rng.randn(4)
                else:
                    policy = random_policy
                    init_state = np.array([0.01, 0.01, np.pi * 0.5, 0.1]) * rng.randn(4)

                # state_traj (aka the x values)
                # state_traj: (50, 4)
                ts, state_traj, action_traj = sim_rollout(sim, policy, NUM_DATAPOINTS_PER_EPOCH, DELTA_T, init_state)
                

                # delta states (aka the y values?)
                # shape: (50, 4)
                delta_state_traj = state_traj[1:] - state_traj[:-1]

                for i in range(len(state_traj) - 1):
                # for i in range(2):
                    if i == 0 and not os.path.exists(PATH + str(epoch) + "/"):
                        os.mkdir(PATH + str(epoch) + "/")
                    vis.set_gt_cartpole_state(state_traj[i][3], state_traj[i][2])
                    vis.set_gt_delta_state_trajectory(ts[:i+1], delta_state_traj[:i+1])

                    # vis_img = vis.draw(redraw=(i==0))
                    vis_img = vis.draw_only_cartpole()
                    vis_img = cv2.resize(vis_img, (128, 128))

                    img_path = PATH + str(epoch) + "/" + str(i) + ".jpg"
                    cv2.imwrite(img_path, vis_img)
                    csv_writer.writerow([delta_state_traj[i,0],delta_state_traj[i,1],delta_state_traj[i,2],delta_state_traj[i,3], img_path])


