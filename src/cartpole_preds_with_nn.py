import numpy as np
import torch

# Global variables
NUM_TRAINING_EPOCHS = 12
NUM_DATAPOINTS_PER_EPOCH = 50
NUM_TRAJ_SAMPLES = 10
DELTA_T = 0.05
rng = np.random.RandomState(12345)

# State representation
# dtheta, dx, theta, x

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

def make_test_data(test_x):
    '''
    takes in a 1x50x128x128 matrix 

    outputs a 46x1x5x128x128
    '''
    final_test_x = None
    for i in range(46):
        one_test_point = test_x[:,i:i+5,:,:].unsqueeze(0)
        if final_test_x is None:
            final_test_x = one_test_point
        else:
            final_test_x = torch.cat((final_test_x, one_test_point), axis=0)
    return final_test_x.float()

def predict_cartpole(test_x, init_state):
    """
    arguments:
    - test_x     : 46x1x5x128x128 matrix to use for the neural networks 
    - init_State : 
    """
    from cartpolenetlite import CartpoleNetLite

    
    M = test_x.shape[0]
    H = NUM_DATAPOINTS_PER_EPOCH
    N = NUM_TRAJ_SAMPLES

    # TODO: Compute these variables.
    pred_gp_mean = np.zeros((NUM_DATAPOINTS_PER_EPOCH, 4))
    pred_gp_variance = np.zeros((NUM_DATAPOINTS_PER_EPOCH, 4))
    rollout_gp = np.zeros((NUM_DATAPOINTS_PER_EPOCH, 4))

    net = CartpoleNetLite()
    # net.load_state_dict(torch.load('./models/CartpoleNet3.pth'))
    # net.eval()
    output = net(test_x)
    print(output.shape)

    pred_gp_mean[4:,:] = output.detach().numpy()

    pred_gp_mean_trajs = np.zeros((NUM_TRAJ_SAMPLES, NUM_DATAPOINTS_PER_EPOCH, 4))
    pred_gp_variance_trajs = np.zeros((NUM_TRAJ_SAMPLES, NUM_DATAPOINTS_PER_EPOCH, 4))
    rollout_gp_trajs = np.zeros((NUM_TRAJ_SAMPLES, NUM_DATAPOINTS_PER_EPOCH, 4))

    return pred_gp_mean, pred_gp_variance, rollout_gp, pred_gp_mean_trajs, pred_gp_variance_trajs, rollout_gp_trajs

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    from cartpole_sim import CartpoleSim
    from policy import SwingUpAndBalancePolicy, RandomPolicy
    from visualization import Visualizer
    import cv2

    vis = Visualizer(cartpole_length=1.5, x_lim=(0.0, DELTA_T * NUM_DATAPOINTS_PER_EPOCH))
    vis2 = Visualizer(cartpole_length=1.5, x_lim=(0.0, DELTA_T * NUM_DATAPOINTS_PER_EPOCH)) # use to create test_x
    
    swingup_policy = SwingUpAndBalancePolicy('policy.npz')
    random_policy = RandomPolicy(seed=12831)
    sim = CartpoleSim(dt=DELTA_T)

    # Initial training data used to train GP for the first epoch
    init_state = np.array([0.01, 0.01, np.pi * 0.5, 0.1]) * rng.randn(4)
    ts, state_traj, action_traj = sim_rollout(sim, random_policy, NUM_DATAPOINTS_PER_EPOCH, DELTA_T, init_state)
    delta_state_traj = state_traj[1:] - state_traj[:-1]
    test_x = None

    for epoch in range(NUM_TRAINING_EPOCHS):
        vis.clear()
        vis2.clear()
        # Use learned policy every 4th epoch
        if (epoch + 1) % 4 == 0:
            policy = swingup_policy
            init_state = np.array([0.01, 0.01, 0.05, 0.05]) * rng.randn(4)
        else:
            policy = random_policy
            init_state = np.array([0.01, 0.01, np.pi * 0.5, 0.1]) * rng.randn(4)

        ts, state_traj, action_traj = sim_rollout(sim, policy, NUM_DATAPOINTS_PER_EPOCH, DELTA_T, init_state)
        delta_state_traj = state_traj[1:] - state_traj[:-1]

        test_x = None
        for i in range(len(state_traj) - 1):
            vis2.set_gt_cartpole_state(state_traj[i][3], state_traj[i][2])
            vis2.set_gt_delta_state_trajectory(ts[:i+1], delta_state_traj[:i+1])

            vis_img = vis2.draw_only_cartpole()
            vis_img = cv2.resize(vis_img, (128, 128))
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)
            cv2.imshow('vis', vis_img)
            # cv2.waitKey(0)
            vis_img = torch.Tensor(vis_img).unsqueeze(0).unsqueeze(0)
            if test_x == None:
                test_x = vis_img
            else:
                test_x = torch.cat((test_x, vis_img), axis=1)

        test_x = make_test_data(test_x)

        (pred_gp_mean,
         pred_gp_variance,
         rollout_gp,
         pred_gp_mean_trajs,
         pred_gp_variance_trajs,
         rollout_gp_trajs) = predict_cartpole(test_x, init_state)

        for i in range(len(state_traj) - 1):
            vis.set_gt_cartpole_state(state_traj[i][3], state_traj[i][2])
            vis.set_gt_delta_state_trajectory(ts[:i+1], delta_state_traj[:i+1])

            if i == 0:
                vis.set_gp_cartpole_state(state_traj[i][3], state_traj[i][2])
                vis.set_gp_cartpole_rollout_state([state_traj[i][3]] * NUM_TRAJ_SAMPLES,
                                                  [state_traj[i][2]] * NUM_TRAJ_SAMPLES)
            else:
                vis.set_gp_cartpole_state(rollout_gp[i-1][3], rollout_gp[i-1][2])
                vis.set_gp_cartpole_rollout_state(rollout_gp_trajs[:, i-1, 3], rollout_gp_trajs[:, i-1, 2])

            vis.set_gp_delta_state_trajectory(ts[:i+1], pred_gp_mean[:i+1], pred_gp_variance[:i+1])

            if policy == swingup_policy:
                policy_type = 'swing up'
            else:
                policy_type = 'random'

            vis.set_info_text('epoch: %d\npolicy: %s' % (epoch, policy_type))

            vis_img = vis.draw(redraw=(i==0))
            cv2.imshow('vis', vis_img)

            if epoch == 0 and i == 0:
                # First frame
                video_out = cv2.VideoWriter('cartpole.mp4',
                                            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                            int(1.0 / DELTA_T),
                                            (vis_img.shape[1], vis_img.shape[0]))

            video_out.write(vis_img)
            cv2.waitKey(int(1000 * DELTA_T))