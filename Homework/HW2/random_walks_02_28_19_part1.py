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
        theta_s_array = [round(math.pi / 24, 4), round(math.pi / 12, 4), round(math.pi / 3, 4)]  # the width of the random walk turning angle distribution (the lower it is, the more straight the trajectory will be)
        w_array = [0.0, 0.5, 1.0]  # w is the weighting given to the directional bias (and hence (1-w) is the weighting given to correlated motion)
        ratio_theta_s_brw_crw = 1
        plot_walks = 0
        count = 0

        efficiency_array = np.zeros([len(theta_s_array), len(w_array), N])
        for w_i in range(len(w_array)):
            w = w_array[w_i]
            for theta_s_i in range(len(theta_s_array)):
                theta_s_crw = np.multiply(ratio_theta_s_brw_crw, theta_s_array[theta_s_i])
                theta_s_brw = theta_s_array[theta_s_i]
                x, y = self.BCRW(N, realizations, v, theta_s_crw, theta_s_brw, w)
                if plot_walks == 1:
                    count += 1
                    plt.figure(count)
                    plt.plot(x.T, y.T)
                    plt.axis('equal')
                efficiency_over_time_steps = np.zeros(500)
                for i in range(N):
                    efficiency_over_time_steps[i] = np.divide(np.mean(x[:, i] - x[:, 0]), (v * i))
                efficiency_array[theta_s_i, w_i] = efficiency_over_time_steps
            # plt.show()
        plt.figure()
        legend_array = []
        w_array_i = np.repeat(w_array, len(efficiency_array))
        for theta_s_i in range(0, len(theta_s_array)):
            for w_i in range(len(w_array)):
                legend_array.append(
                    ["$\theta^{*CRW}=$", (ratio_theta_s_brw_crw * theta_s_array[theta_s_i]),
                     "$\theta^{*BRW}=$", (theta_s_array[theta_s_i]),
                     "$w=$", w_array[w_i]]
                )

        plt.title('Navigational Efficiency vs. Time')
        plt.xlabel('Time Step')
        plt.ylabel('Navigational Efficiency')
        colors = ['b', 'g', 'r', 'b', 'g', 'r', 'b', 'g', 'r']
        linestyles = ['solid', 'solid', 'solid', 'dotted', 'dotted', 'dotted', 'dashed', 'dashed', 'dashed']
        for efficiency, color, label, linestyle in zip(
                                                        np.reshape(efficiency_array,(len(theta_s_array)*len(w_array), N)),
                                                        colors,
                                                        legend_array,
                                                        linestyles):
            plt.plot(range(N), efficiency, color, label=label, linestyle=linestyle, alpha=.75)
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
