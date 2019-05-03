import random
import math

class StoppingProblem:
    def __init__(self, length=100, upper_bound=1000, seed=None):
        if seed is not None:
            random.seed(seed)
        self.upper_bound = upper_bound
        self._numbers = [random.randint(1, upper_bound) for x in range(length)]
        self._index = 0

    def peek(self):
        if self._index + 1 < len(self):
            return self._numbers[self._index + 1]
        else:
            return 0

    def highest(self, value):
        return value >= max(self._numbers[:self._index+1])
    
    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self):
            self._index = self._index + 1
            return self._numbers[self._index - 1]
        else:
            self._index = 0
            raise StopIteration

    def __len__(self):
        return len(self._numbers)

    def __max__(self):
        return self.upper_bound
        

class StoppingProblemAgent:
    def __init__(self):
        pass

    def Reward(self, stopping_problem):
        pass
    
class HeuristicAgent(StoppingProblemAgent):
    heuristic_prop = math.exp(-1)
    def Reward(self, stopping_problem):
        stopping_point = math.floor((len(stopping_problem)*self.heuristic_prop))
        prev_max = float("-Infinity")
        problem_iter = iter(stopping_problem)
        i = 1
        reward = float("-Infinity")
        while i < stopping_point:
            reward = next(problem_iter)
            if reward > prev_max:
                prev_max = reward  
            i = i + 1
        while True:
            try:
                reward = next(problem_iter)
                last_reward = reward
                if reward > prev_max:
                    return reward, i
                i = i + 1
            except StopIteration:
                return reward, i - 1
            

class QValueAgent(StoppingProblemAgent):
    def __init__(self, explore_rate = 0.1, value_precision=0.01, step_precision=0.1):
        self.explore_rate = explore_rate
        self.Q_Table = [[[0, 1] for value in range(math.floor(1/value_precision))]
                        for index in range(math.floor(1/step_precision))]
        self.value_precision = value_precision
        self.step_precision = step_precision

    def GetQ(self, index_f, value_f, choice=None):
        index_f = max(0, min(index_f, 1 - self.step_precision))
        value_f = max(0, min(value_f, 1 - self.value_precision))
        index = math.floor(index_f/self.step_precision)
        value = math.floor(value_f/self.value_precision) - 1
        if choice is not None:
            return self.Q_Table[index][value][choice]
        else:
            return self.Q_Table[index][value]

    def SetQ(self, index_f, value_f, choice, newQ):
        index_f = max(0, min(index_f, 1))
        value_f = max(0, min(value_f, 1))
        index = math.floor(index_f/self.step_precision)
        value = math.floor(value_f/self.value_precision) - 1
        self.Q_Table[index][value][choice] = newQ

    def Policy(self, index_f, value_f, exploration=False):
        if index_f >= 1:
            return 0
        stayQ, nextQ = self.GetQ(index_f, value_f)
        if exploration and random.random() < self.explore_rate:
            if stayQ > nextQ:
                return 1
            else:
                return 0
        else:
            if stayQ > nextQ:
                return 0
            else:
                return 1
        
    def Train(self, steps=1000, learning_rate=0.1, discount=0.9, length=100, upper_bound=1000):
        rewards = []
        for x in range(steps):
            stopping_problem = StoppingProblem(length=length, upper_bound=upper_bound, seed = (x % int(steps/20)))
            for index, reward in enumerate(stopping_problem):
                index_f = index/len(stopping_problem)
                value_f = reward/stopping_problem.upper_bound
                policy = self.Policy(index_f, value_f, True)
                    
                if policy == 1:
                    td_target = discount * max(self.GetQ((index+1)/len(stopping_problem), stopping_problem.peek()/stopping_problem.upper_bound))
                    td_error = td_target - self.GetQ(index_f, value_f, policy)
                    self.SetQ(index_f, value_f, policy, self.GetQ(index_f, value_f, policy) + learning_rate * td_error)
                else:
                    td_target = value_f
                    td_error = td_target - self.GetQ(index_f, value_f, policy)
                    self.SetQ(index_f, value_f, policy, self.GetQ(index_f, value_f, policy) + learning_rate * td_error)
                    rewards.append(reward)
                    break
        return rewards
    
    def Reward(self, stopping_problem):
        last_reward = float("-Infinity")
        for index, reward in enumerate(stopping_problem):
            index_f = index/len(stopping_problem)
            value_f = reward/stopping_problem.upper_bound
            policy = self.Policy(index_f, value_f)
            last_reward = reward
            if policy == 0:
                return reward, index
        if last_reward == float("-Infinity"):
            raise Exception
        return last_reward, len(stopping_problem)
        

class QRankAgent(StoppingProblemAgent):
    def __init__(self, explore_rate = 0.1, step_precision = 0.01):
        self.explore_rate = explore_rate
        self.step_precision = step_precision
        self.Q_Table = [[[0,0],[0,0]] for index in range(math.floor(1/step_precision))]

    def GetQ(self, index_f, highest, choice=None):
        index_f = max(0, min(index_f, 1 - self.step_precision))
        index = math.floor(index_f/self.step_precision)
        if highest:
            highest = 1
        else:
            highest = 0
        if choice is not None:
            return self.Q_Table[index][highest][choice]
        else:
            return self.Q_Table[index][highest]

    def SetQ(self, index_f, highest, choice, newQ):
        index_f = max(0, min(index_f, 1 - self.step_precision))
        index = math.floor(index_f/self.step_precision)
        if highest:
            highest = 1
        else:
            highest = 0
        self.Q_Table[index][highest][choice] = newQ

    def Policy(self, index_f, highest, exploration=False):
        if index_f >= 1:
            return 0
        stayQ, nextQ = self.GetQ(index_f, highest)
        if exploration and random.random() < self.explore_rate:
            if stayQ > nextQ:
                return 1
            else:
                return 0
        else:
            if stayQ > nextQ:
                return 0
            else:
                return 1

    def Train(self, steps=1000, learning_rate=0.1, discount=0.9, length=100, upper_bound=1000):
        rewards = []
        for x in range(steps):
            stopping_problem = StoppingProblem(length=length, upper_bound=upper_bound, seed = (x % int(steps/20)))
            for index, reward in enumerate(stopping_problem._numbers, start=1):
                index_f = index/len(stopping_problem)
                highest = stopping_problem.highest(reward)
                policy = self.Policy(index_f, highest, True)
                if policy == 1:
                    td_target = discount * max(self.GetQ((index+1)/len(stopping_problem), stopping_problem.highest(stopping_problem.peek())))
                    td_error = td_target - self.GetQ(index_f, highest, policy)
                    self.SetQ(index_f, highest, policy, self.GetQ(index_f, highest, policy) + learning_rate * td_error)
                else:
                    td_target = reward
                    td_error = td_target - self.GetQ(index_f, highest, policy)
                    self.SetQ(index_f, highest, policy, self.GetQ(index_f, highest, policy) + learning_rate * td_error)
                    rewards.append(reward)
                    break
        return rewards

    def Reward(self, stopping_problem):
        last_reward = float("-Infinity")
        for index, reward in enumerate(stopping_problem):
            index_f = index/len(stopping_problem)
            highest = stopping_problem.highest(reward)
            policy = self.Policy(index_f, highest)
            last_reward = reward
            if policy == 0:
                return reward, index
        if last_reward == float("-Infinity"):
            raise Exception
        return last_reward, len(stopping_problem)


if __name__ == "__main__":
    pass
        
        
        
