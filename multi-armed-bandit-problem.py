class EpsilonGreedyArm:
    def __init__(self, epsilon, n_arms):
        self.n_arms = n_arms
        # number of times each arm was chosen
        self.counts = np.zeros(n_arms)
        # average score of each arm
        self.values = np.zeros(n_arms)
        self.epsilon = epsilon

    def select_arm(self):
        if random.random() > self.epsilon:
            return np.argmax(self.values)
        else:
            return random.randint(0, self.n_arms - 1)
 
    def update(self, chosen_arm, reward):
        n = self.counts[chosen_arm]
        average = self.values[chosen_arm]
        # get a sum from an average and calculate the new average
        new_average = (average * n + reward) / (n + 1)
        self.values[chosen_arm] = new_average
        self.counts[chosen_arm] += 1

class UCBArm(): 
    def __init__(self, n_arms):
        self.n_arms = n_arms
        # number of times each arm was chosen
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
 
    def select_arm(self):
        if min(self.counts) == 0:
            return np.argmin(self.counts)
 
        total_counts = sum(self.counts)
        bonus = np.sqrt(2 * np.log(total_counts) / self.counts)
        ucb_values = self.values + bonus
        return np.argmax(ucb_values)
 
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value

class ThompsonSamplingArm():
    def __init__(self, n_arms):
        # a paramter of beta distribution
        self.counts_alpha = np.zeros(n_arms)
        # a paramter of beta distribution
        self.counts_beta = np.zeros(n_arms)
        # fixed value
        self.alpha = 1
        # fixed value
        self.beta = 1
        self.n_arms = n_arms
 
    def __get_beta_distribution(self, arm):
        return random.betavariate(self.counts_alpha[arm] + self.alpha,
                                self.counts_beta[arm] + self.beta)
    
    def select_arm(self):
        betas = [self.__get_beta_distribution(arm) for arm in range(self.n_arms)]
        return np.argmax(betas)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value

        if new_value > 0:
            self.alpha[chosen_arm] += 1
        else:
            self.beta[chosen_arm] += 1
