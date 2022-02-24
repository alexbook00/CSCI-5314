import matplotlib.pyplot as plt
import math
import numpy as np
import random
import pylab


class random_walks_python():
    def random_walks(self):
        N = 500  # no of steps per trajectory
        realizations = 50  # number of trajectories
        v = 1.0  # velocity (step size)
        theta_s_array = [round(math.pi / 3, 4)]  # the width of the random walk turning angle distribution (the lower it is, the more straight the trajectory will be)
        w_array = np.arange(0, 1.001, .001)  # w is the weighting given to the directional bias (and hence (1-w) is the weighting given to correlated motion)
        ratio_theta_s_brw_crw = .1
        plot_walks = 0
        count = 0

        efficiency_array = np.zeros(len(w_array))
        for w_i in range(len(w_array)):
            w = w_array[w_i]
            print(w)
            for theta_s_i in range(len(theta_s_array)):
                theta_s_crw = np.multiply(ratio_theta_s_brw_crw, theta_s_array[theta_s_i])
                theta_s_brw = theta_s_array[theta_s_i]
                x, y = self.BCRW(N, realizations, v, theta_s_crw, theta_s_brw, w)
                if plot_walks == 1:
                    count += 1
                    plt.figure(count)
                    plt.plot(x.T, y.T)
                    plt.axis('equal')
                efficiency = np.divide(np.mean(x[:, -1] - x[:, 0]), (v * N))
                efficiency_array[w_i] = efficiency
            # plt.show()

        plt.figure()

        plt.title("Navigational Efficiency vs. $w$")
        plt.xlabel("$w$ (directional bias weighting)")
        plt.ylabel('Navigational Efficiency')
        plt.plot(w_array, efficiency_array, 'b', alpha=.75)
        plt.legend(loc='lower right', prop={'size': 8})
        plt.show()

    # The function generates 2D-biased correlated random walks
    def BCRW(self, N, realizations, v, theta_s_crw, theta_s_brw, w):
        X = np.zeros([realizations, N])
        Y = np.zeros([realizations, N])
        theta = np.zeros([realizations, N])
        X[:, 0] = 0
        Y[:, 0] = 0
        theta[:, 0] = 0

        for realization_i in range(realizations):
            for step_i in range(1, N):
                theta_crw = theta[realization_i][step_i - 1] + (theta_s_crw * 2.0 * (np.random.rand(1, 1) - 0.5))
                theta_brw = (theta_s_brw * 2.0 * (np.random.rand(1, 1) - 0.5))

                X[realization_i, step_i] = X[realization_i][step_i - 1] + (v * (w * math.cos(theta_brw))) + (
                            (1 - w) * math.cos(theta_crw))
                Y[realization_i, step_i] = Y[realization_i][step_i - 1] + (v * (w * math.sin(theta_brw))) + (
                            (1 - w) * math.sin(theta_crw))

                current_x_disp = X[realization_i][step_i] - X[realization_i][step_i - 1]
                current_y_disp = Y[realization_i][step_i] - Y[realization_i][step_i - 1]
                current_direction = math.atan2(current_y_disp, current_x_disp)

                theta[realization_i, step_i] = current_direction

        return X, Y


rdm_plt = random_walks_python()
rdm_plt.random_walks()
