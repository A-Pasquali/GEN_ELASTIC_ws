import numpy as np
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.mplot3d import Axes3D

class Individual:
    def __init__(self, A, B, points = [], number_of_points = 20, Kpoint = [5,5,5], K = 1, sigma = 1):
        self.A = A #start point
        self.B = B #end point
        self.number_of_points = number_of_points
        if len(points) == 0:
            self.init_points()
        else:
            self.points = points
        self.Kpoint = Kpoint
        self.K = K
        self.sigma = sigma
        self.score = self.fitness()

    def init_points(self): #linear interpolation between A and B
        self.points = np.zeros((self.number_of_points, self.A.shape[0]))
        for i in range(1, self.number_of_points+1):
            self.points[i-1, :] = self.A + i/(self.number_of_points+1) * (self.B - self.A)

    def distance(self, p1, p2):
        return np.linalg.norm(p1-p2)
    
    def force(self, p):
        return self.K * self.distance(p, self.Kpoint)

    def fitness(self):
        distAB = self.distance(self.A, self.B)
        max_force = 30
        correct_distance = distAB/(self.number_of_points+1)

        path = np.vstack((self.A.reshape(1,3), self.points, self.B.reshape(1,3)))
        distances = []

        for i in range(1, len(path)-1):
            prev_dist = self.distance(path[i-1, :], path[i, :])
            post_dist = self.distance(path[i, :], path[i+1, :])
            distances.append((1+((0.5*abs(prev_dist - correct_distance) + 0.5*abs(post_dist - correct_distance))/distAB))**2)

        forces = []
        for i in range(1,len(path)-1):
            forces.append((1+self.force(path[i, :])/max_force))

        points_scores = []
        for i in range(len(path)-2):
            points_scores.append(forces[i]*distances[i])

        score = np.sum(points_scores)

        self.mutation_prob = np.array(points_scores)/np.sum(points_scores)

        """
        distances = [1]
        distances.append(1+(abs(self.distance(self.A, self.points[0, :])-correct_distance)))
        for i in range(self.number_of_points-1):
            distances.append(1+(abs(self.distance(self.points[i, :], self.points[i+1, :])-correct_distance)))
        distances.append(1+(abs(self.distance(self.points[-1, :], self.B)-correct_distance)))

        forces = []
        forces.append(1+self.force(self.A)/max_force)
        for i in range(self.number_of_points):
            forces.append(1+self.force(self.points[i, :])/max_force)
        forces.append(1+self.force(self.B)/max_force)

        points_scores = []
        for i in range(self.number_of_points+2):
            points_scores.append(0.25*forces[i]*(distances[i])**2)
        score = np.sum(points_scores)

        self.mutation_prob = np.array(points_scores[1:-1])/np.sum(points_scores[1:-1])
        """
        return score
    
    def big_mutation(self):
        #mute all points
        points = copy.deepcopy(self.points)
        #points += np.random.randint(-self.sigma, self.sigma, size = (self.number_of_points, self.A.shape[0]))
        points += np.round(2*(0.5 - np.random.rand(self.number_of_points, self.A.shape[0]))*self.sigma,3)
        return Individual(self.A, self.B, points, self.number_of_points, self.Kpoint, self.K, self.sigma)
    
    def mutate(self):
        #select random point in self.points
        points = copy.deepcopy(self.points)
        index = np.random.choice(np.arange(self.number_of_points), p = self.mutation_prob)
        #mutate
        points[index, :] += np.round(2*(0.5 - np.random.rand(self.A.shape[0]))*self.sigma, 3)
        return Individual(self.A, self.B, points, self.number_of_points, self.Kpoint, self.K, self.sigma)

    def __mul__(self, other):
        #select random between self and other for each point
        points = np.zeros((self.number_of_points, self.A.shape[0]))
        for i in range(self.number_of_points):
            if self.mutation_prob[i] < other.mutation_prob[i]:
                points[i, :] = self.points[i, :]
            else:
                points[i, :] = other.points[i, :]
        return Individual(self.A, self.B, points, self.number_of_points, self.Kpoint, self.K, self.sigma)


class Population:
    def __init__(self, number_individuals, A, B, number_of_points = 20, Kpoint = [5,5,5], K = 10, sigma = 10, delta = "default"):
        self.number_individuals = number_individuals
        self.A = A
        self.B = B
        self.number_of_points = number_of_points
        self.Kpoint = Kpoint
        self.K = K
        self.sigma = sigma
        self.delta = delta
        self.individuals = []
        self.init_population()

    def init_population(self):
        first = Individual(self.A, self.B, number_of_points = self.number_of_points, Kpoint = self.Kpoint, K = self.K, sigma = self.sigma)
        #self.individuals.append(first)
        for i in range(self.number_individuals):
            self.individuals.append(first.big_mutation())

    def get_scores(self):
        scores = []
        for individual in self.individuals:
            scores.append(individual.score)
        self.scores = np.array(scores)

    def reorder(self):
        sorted_score = copy.deepcopy(self.scores)
        sorted_score.sort()
        new_individuals = []
        for i in range(len(sorted_score)):
            index = np.where(self.scores == sorted_score[i])[0][0]
            new_individuals.append(self.individuals[index])
        self.individuals = new_individuals
        self.scores = sorted_score

    def fit(self, number_generations, Cw = 1/2, Cm = 1/2, Ccr = 0):
        self.paths_tested = []
        self.scores_obtained = []
        if self.delta == "default":
            self.delta = (0.01/self.sigma)**(1/number_generations)
            print("Delta: ", self.delta)
            print("Sigma: ", self.sigma)
        for i in range(number_generations):
            self.sigma *= self.delta
            #self.sigma = round(self.sigma)
            self.get_scores()
            self.reorder()
            scores = []
            for individual in self.individuals[:int(self.number_individuals*Cw)]:
                individual.sigma = self.sigma
                self.paths_tested.append(individual.points)
                scores.append(individual.score)
            self.scores_obtained.append(copy.deepcopy(scores))
            #print("Generation: ", i, " Score: ", self.scores[0])
            self.winners = self.individuals[:int(self.number_individuals*Cw)]
            self.new_individuals = []
            self.mutate(Cm)
            self.reproduce(Ccr)
            self.individuals = self.winners + self.new_individuals
        self.reorder()
        score = []
        for individual in self.individuals[:int(self.number_individuals*Cw)]:
            self.paths_tested.append(individual.points)
            score.append(individual.score)
        self.scores_obtained.append(copy.deepcopy(score))
        self.best_path = self.individuals[0].points

    def mutate(self, Cm):
        for i in range(int(self.number_individuals*Cm)):
            self.new_individuals.append(self.winners[i].mutate())
        
    def reproduce(self, Ccr):
        for i in range(int(self.number_individuals*Ccr)):
            random_index = i
            while random_index == i:
                random_index = np.random.randint(0, len(self.winners))
            self.new_individuals.append(self.winners[i]*self.winners[random_index])

class ESlearning:
    def __init__(self, Parameters : dict):
        self.A = Parameters["A"]
        self.B = Parameters["B"]
        self.number_of_points = Parameters["number_of_points"]
        self.Kpoint = Parameters["Kpoint"]
        self.K = Parameters["K"]
        self.number_individuals = Parameters["number_individuals"]
        self.Cw = Parameters["Cw"]
        self.Cm = Parameters["Cm"]
        self.Ccr = Parameters["Ccr"]
        self.Rep = Parameters["Rep"]
        self.Gen = Parameters["Gen"]
        self.end_sigma = Parameters["end_sigma"]
        self.colors = Parameters["colors"]
        self.Sigma = np.linalg.norm(self.A-self.B)/(self.number_of_points+1)
        self.Dict_results = {}
        self.start_name = "Experiment_"
        self.save_dir_img = "/home/alex/ros/GEN_ELASTIC_ws/scripts/images/"
        self.save_dir_data = "/home/alex/ros/GEN_ELASTIC_ws/scripts/data/"
        self.label_used = []

    def change_Dict_colors(self):
        for Gen, Cw, Cm, Ccr, Ni, Col in zip(self.Gen, self.Cw, self.Cm, self.Ccr, self.number_individuals, self.colors):
            for K in self.K:
                key = self.start_name + "Ni" + str(Ni) + "_Gen" + str(Gen) + "_Cw" + str(Cw) + "_Cm" + str(Cm) + "_Ccr" + str(Ccr) + "_K" + str(K)
                self.Dict_results[key]["Color"] = Col

    def save_data(self):
        np.save(self.save_dir_data + self.start_name + "Dict_results.npy", self.Dict_results)

    def load_data(self):
        self.Dict_results = np.load(self.save_dir_data + self.start_name + "Dict_results.npy", allow_pickle = True).item()
    
    def perform_experiment(self, Gen, Cw, Cm, Ccr, Ni, col):
        self.delta = (self.end_sigma/self.Sigma)**(1/Gen)
        for K in self.K:
            score_rep = []
            for rep in range(self.Rep):
                Pop = Population(Ni, self.A, self.B, number_of_points = self.number_of_points, Kpoint = self.Kpoint, K = K, sigma = self.Sigma, delta = self.delta)
                Pop.fit(Gen, Cw = Cw, Cm = Cm, Ccr = Ccr)
                Pop.scores_obtained = np.array(Pop.scores_obtained)
                score_rep.append(Pop.scores_obtained[:, 0])
            means = np.mean(score_rep, axis = 0)
            stds = np.std(score_rep, axis = 0)
            self.Dict_results[self.start_name + "Ni" + str(Ni) + "_Gen" + str(Gen) + "_Cw" + str(Cw) + "_Cm" + str(Cm) + "_Ccr" + str(Ccr) + "_K" + str(K)] = {"means" : means, "stds" : stds, "Ni" : Ni, "Gen" : Gen, "Cw" : Cw, "Cm" : Cm, "Ccr" : Ccr, "K" : K, "Rep" : self.Rep, "Color" : col}
        self.save_data()

    def perform_all_experiments(self):
        counter_experiment = 1
        for Gen, Cw, Cm, Ccr, Ni, Col in zip(self.Gen, self.Cw, self.Cm, self.Ccr, self.number_individuals, self.colors):
            print("Experiment number: ", counter_experiment, " out of ", len(self.Gen))
            self.perform_experiment(Gen, Cw, Cm, Ccr, Ni, Col)       
            counter_experiment += 1 

    def plot_mean_std_results(self, Gen, Cw, Cm, Ccr, Ni, K):
        label = "Gen = " + str(Gen) + ", Cw = " + str(Cw) + ", Cm = " + str(Cm) + ", Ccr = " + str(Ccr)
        plt.xlabel("Generation", fontsize = 12)
        plt.ylabel("Score", fontsize = 12)
        key = self.start_name + "Ni" + str(Ni) + "_Gen" + str(Gen) + "_Cw" + str(Cw) + "_Cm" + str(Cm) + "_Ccr" + str(Ccr) + "_K" + str(K)
        if label not in self.label_used:
            self.label_used.append(label)
            plt.plot(np.arange(Gen+1), self.Dict_results[key]["means"], linewidth = 2, label = label, color = self.Dict_results[key]["Color"])
        else:
            plt.plot(np.arange(Gen+1), self.Dict_results[key]["means"], linewidth = 2, color = self.Dict_results[key]["Color"])
        plt.fill_between(np.arange(Gen+1), self.Dict_results[key]["means"] - self.Dict_results[key]["stds"], self.Dict_results[key]["means"] + self.Dict_results[key]["stds"], alpha = 0.1, color = self.Dict_results[key]["Color"])
        current_min = np.min(self.Dict_results[key]["means"])
        color_min = self.Dict_results[key]["Color"]
        return current_min, color_min
    
    def plot_results_of_one_Ni(self, Ni):
        plt.figure()
        plt.title("Comparison of different Cw, Cm, Ccr with a Population of " + str(Ni) + " Individuals", fontsize = 14)
        Ni_possible = np.array(self.number_individuals)
        indexes = np.where(Ni_possible == Ni)[0]
        min_values = {str(self.K[0]): [], str(self.K[1]): [], str(self.K[2]): []}
        col_min = {str(self.K[0]): [], str(self.K[1]): [], str(self.K[2]): []}
        for index in indexes:
            Gen = self.Gen[index]
            Cw = self.Cw[index]
            Cm = self.Cm[index]
            Ccr = self.Ccr[index]
            plt.xlabel("Generation", fontsize = 12)
            plt.ylabel("Score", fontsize = 12)
            for K in self.K:
                minimum, col = self.plot_mean_std_results(Gen, Cw, Cm, Ccr, Ni, K)
                min_values[str(K)].append(minimum)
                col_min[str(K)].append(col)
        for key in min_values.keys():
            min_value = np.min(min_values[key])
            index = np.where(min_values[key] == min_value)[0][0]
            col = col_min[key][index]
            plt.plot(np.arange(Gen+1), np.ones(Gen+1)*min_value, '--', linewidth = 2, label = "Minimum Value = " + str(round(min_value, 3)) + " with K = " + key, alpha = 0.5, color = col)
        plt.legend()
        #aumentare la grandezza del grafico
        plt.gcf().set_size_inches(8, 7)
        plt.savefig(self.save_dir_img + self.start_name + "Ni" + str(Ni) + ".png")
        plt.show()

    def plot_results(self):
        different_Ni = np.unique(self.number_individuals)
        for Ni in different_Ni:
            self.plot_results_of_one_Ni(Ni)

if __name__ == "__main__":

    Parameters = {"A" : np.array([0, 0, 0]), 
                "B" : np.array([0.6, 0.6, 0.6]), 
                "number_of_points" : 5, 
                "Kpoint" : [0.6,0,0], 
                "K" : [10,20,30], 
                "number_individuals" : [8,8,8,8,16,16,16,16,32,32,32,32,64,64,64,64], 
                "Cw" : [1/2,1/2,3/4,3/4,1/2,1/2,3/4,3/4,3/4,3/4,7/8,7/8,7/8,7/8,15/16,15/16], 
                "Cm" : [1/2,1/4,2/8,1/8,1/2,6/16,2/16,3/16,2/16,3/16,2/32,2/32,4/64,6/64,2/64,3/64], 
                "Ccr" : [0,1/4,0,1/8,0,2/16,2/16,1/16,2/16,1/16,2/32,1/32,4/64,2/64,2/64,1/64],
                "Rep" : 50, 
                "Gen" : [123,123,246,246,61,61,121,121,58,58,117,117,54,54,109,109],
                "end_sigma" : 0.001,
                "colors" : ["red", "blue", "green", "purple", "red", "blue", "green", "purple", "red", "blue", "green", "purple", "red", "blue", "green", "purple"]
                }

    ES = ESlearning(Parameters)
    ES.perform_all_experiments()
    ES.plot_results()

# #test individual
# A = np.array([0, 0, 0])
# B = np.array([0.6, 0.6, 0.6])

# distAB = np.linalg.norm(A-B)

# number_of_points = 5

# Pop = Population(16, A, B, number_of_points = number_of_points, Kpoint = [0.6,0,0], K = 10, sigma = distAB/(number_of_points+1))
# Pop.fit(123, Cw = 3/4, Cm = 3/16, Ccr = 1/16)

# Kpoint = Pop.Kpoint

# #3d plot with axis3d
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.scatter(A[0], A[1], A[2], c='g', marker='o', s = 100)
# ax.scatter(B[0], B[1], B[2], c='g', marker='o', s = 100)
# start_color = np.array([0, 0, 1])
# end_color = np.array([0, 1, 0])
# for i,path in enumerate(Pop.paths_tested):
#     path = np.vstack((A.reshape(1,3), path, B.reshape(1,3)))
#     color = start_color + (end_color - start_color) * i/len(Pop.paths_tested)
#     ax.plot(path[:, 0], path[:, 1], path[:, 2], c=color, alpha = 0.5, linewidth = 0.1)
# best_path = np.vstack((A.reshape(1,3), Pop.best_path, B.reshape(1,3)))
# ax.plot(best_path[:, 0], best_path[:, 1], best_path[:, 2], c='g', alpha = 1, linewidth = 3)
# ax.scatter(best_path[:, 0], best_path[:, 1], best_path[:, 2], c='g', marker='o', s = 100)
# ax.scatter(Kpoint[0], Kpoint[1], Kpoint[2], c='purple', marker='*', s = 100)
# ax.xlim = (0, 600)
# ax.ylim = (0, 600)
# ax.zlim = (0, 600)
# #trait lines between best points and Kpoint
# for i in range(len(Pop.best_path)):
#     ax.plot([Pop.best_path[i, 0], Kpoint[0]], [Pop.best_path[i, 1], Kpoint[1]], [Pop.best_path[i, 2], Kpoint[2]], c='purple', alpha = 0.5, linewidth = 2, linestyle = '--')
# plt.show()

# Pop.scores_obtained = np.array(Pop.scores_obtained)

# #score plot, riempire l'area compresa tra le due curve Pop.scores_obtained[0, :] e Pop.scores_obtained[-1, :]
# plt.figure()
# #area between the two curves
# plt.fill_between(np.arange(len(Pop.scores_obtained[:, 0])), Pop.scores_obtained[:, 0], Pop.scores_obtained[:, -1], color = 'lightblue')
# #plot of the two curves
# plt.plot(np.arange(len(Pop.scores_obtained[:, 0])), Pop.scores_obtained[:, 0], label = 'Best score', color = 'green')
# plt.plot(np.arange(len(Pop.scores_obtained[:, 0])), Pop.scores_obtained[:, -1], label = 'Worst score', color = 'red')
# plt.legend()
# plt.show() 