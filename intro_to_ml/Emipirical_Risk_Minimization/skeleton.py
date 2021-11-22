#################################
#
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        X = np.sort(np.random.uniform(size=m))
        i1 = (X > 0.2) & (X < 0.4)
        i2 = (X > 0.6) & (X < 0.8)
        X_loc = []
        for j in range(m):
            if (not i1[j]) & (not i2[j]):
                X_loc.append(True)
            else:
                X_loc.append(False)
        Y = np.array([self.get_sample(X_loc[i]) for i in range(m)]).reshape(m,)
        return np.column_stack((X, Y))


    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        sample_sizes = np.arange(m_first, m_last + 1, step)
        n_steps = sample_sizes.shape[0]
        empirical_errors = np.zeros(n_steps)
        true_errors = np.zeros(n_steps)
        for i in range(n_steps):
            emp_err = 0
            true_err = 0
            for j in range(T):
                sample = self.sample_from_D(sample_sizes[i])
                xs = sample[:, 0]
                ys = sample[:, 1]
                ERM_intervals, error_count = intervals.find_best_interval(xs, ys, k)
                emp_err += error_count
                true_err += self.true_error(ERM_intervals)
            #find_best_interval returns error count hence the divison by the sample_sizes
            empirical_errors[i] = emp_err/(T * sample_sizes[i])
            true_errors[i] = true_err/(T)
        
        
        plt.plot(sample_sizes, empirical_errors, label="Empirical Error")
        plt.plot(sample_sizes, true_errors, label="True Error")
        plt.legend()
        plt.xlabel("Sample Sizes")
        plt.ylabel("Error")
        plt.show()

        return np.stack((empirical_errors, true_errors))


    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        
        K = np.arange(k_first, k_last + 1, step)
        true_errors = np.zeros(K.shape[0])
        empirical_errors = np.zeros(K.shape[0])
        sample = self.sample_from_D(m)
        xs = sample[:,0]
        ys = sample[:,1]
        for i in range(K.shape[0]):
            ERM_intervals, error_count = intervals.find_best_interval(xs, ys, K[i])
            true_errors[i] = self.true_error(ERM_intervals)
            empirical_errors[i] = error_count/m

        plt.plot(K, true_errors, label="True Error")
        plt.plot(K, empirical_errors, label="Empirical Error")
        plt.legend()
        plt.xlabel("Number of Intervals")
        plt.ylabel("Error")
        plt.show()

        return K[np.argmin(empirical_errors)]
         

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        K = np.arange(k_first, k_last + 1, step)
        true_errors = np.zeros(K.shape[0])
        empirical_errors = np.zeros(K.shape[0])
        sample = self.sample_from_D(m)
        xs = sample[:,0]
        ys = sample[:,1]
        penalties = self.get_penalties(K, m)
        for i in range(K.shape[0]):
            ERM_intervals, error_count = intervals.find_best_interval(xs, ys, K[i])
            true_errors[i] = self.true_error(ERM_intervals)
            empirical_errors[i] = error_count/m
        penalties_with_empirical_errors = penalties + empirical_errors

        plt.plot(K, empirical_errors, label="Empirical Error")
        plt.plot(K, true_errors, label="True Error")
        plt.plot(K, penalties_with_empirical_errors, label="Penalty & Emprical Error")
        plt.plot(K, penalties, label="Penalty")
        plt.legend()
        plt.xlabel("k")
        plt.show()

        return K[np.argmin(penalties_with_empirical_errors)]

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        
        K = []
        sample = self.sample_from_D(m)
        np.random.shuffle(sample)
        validate = sample[:int(m/5)]
        train = np.array(sorted(sample[int(m/5):], key=lambda x:x[0]))
        xt = train[:,0]
        yt = train[:,1]
        xv = validate[:,0]
        yv = validate[:,1]
        for k in range(1,11):
            ERM_k_intervals, error_count = intervals.find_best_interval(xt, yt, k)
            K.append(self.get_validation_error(ERM_k_intervals, xv, yv))
       
        return np.argmin(K) + 1

    #################################
    # Place for additional methods
    def get_rest_of_intervals(self, intervals):
        rest = [(0.0, intervals[0][0])] #Either (0,0) or (0, lower bound of intervals)
        for i in range(len(intervals) - 1):
            rest.append((intervals[i][1], intervals[i+1][0]))
        rest.append((intervals[-1][1], 1.0)) #Either (1,1) or (upper bound of intervals, 1)
        return rest

    def true_error(self, intervals1):
        likely0 = [(0.2, 0.4), (0.6, 0.8)]
        likely1 = [(0, 0.2), (0.4, 0.6), (0.8,1)]
        p0 = 0.1
        p1 = 0.8
        intervals0 = self.get_rest_of_intervals(intervals1)
        result = 0.0

        # Using the law of total expectation, we want to find our when our hypothesis is wrong
        # Expected error when hypothesis returns 1
        for interval in intervals1:
            for seg1 in likely1:
                result += self.get_intersection(interval, seg1) * (1 - p1)
            for seg2 in likely0:
                result += self.get_intersection(interval, seg2) * (1 - p0)
        
        #Expected Error when hypothesis returns 0
        for interval in intervals0:
            for seg1 in likely1:
                result += self.get_intersection(interval, seg1) * p1
            for seg2 in likely0:
                result += self.get_intersection(interval, seg2) * p0
        return result
        
    def get_intersection(self, interval1, interval2):
        # Case1 No intersection
        if (interval1[1] <= interval2[0]) or (interval1[0] >= interval2[1]):
            return 0.0
        #Case 2 + 3: One is a subset of the other
        elif (interval1[0] >= interval2[0]) and (interval1[1] <= interval2[1]):
            return interval1[1] - interval1[0]
        elif (interval1[0] <= interval2[0]) and (interval1[1] >= interval2[1]):
            return interval2[1] - interval2[0]
        #Case 4 + 5: Intersection
        elif (interval1[0] <= interval2[0]) and (interval1[1] <= interval2[1]):
            return interval1[1] - interval2[0]
        else:
            return interval2[1] - interval1[0]

    def get_sample(self, indicator):
        #indicator == True means x belongs to (0, 0.2) or (0.4, 0.6) or (0.8,1)]
        if indicator:
            return np.random.choice(a = [0.0, 1.0], size = 1, p = [0.2, 0.8])
        else:
            return np.random.choice(a = [0.0, 1.0], size = 1, p = [0.9, 0.1])

    def get_penalties(self, K, m):
        #VCdim for Hk is 2k
        ret = (2*K + np.log(20))/m
        return 2*np.sqrt(ret)


    def get_validation_error(self, intervals, x, y):
        error = 0.0
        for i in range(len(x)):
            counter = 0
            for interval in intervals:
               if (interval[0] <= x[i] <= interval[1]):
                   counter += 1
                   break
            if (counter != y[i]):
                error += 1
        return error/len(x)

    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)

