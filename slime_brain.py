#slime brain
#feed forward network where information moves only forward so theres no back propogation 

#actions turn and gulp 
#unlike the flappy bird game
#here we have two independent motions 
#so that will be refelcted in 2 ouputs: 1 is our turn and 1 is gulp 
#libraries
import numpy as np
import random 

class feed_forward_network:

    #here by layers we mean number of hidden layers
    #neurons in each layer mean neurons in each hidden layer they do not count the input and output 
    def __init__(self, layers, no_input, no_output, neurons_in_each_layer):
        self.layers=layers #how many layers
        self.no_input=no_input #number of inputs
        self.no_output=no_output 
        self.neurons_in_each_layer=neurons_in_each_layer
        #total layers
        self.total_layers=layers+2
        #size of every layer in order: input, hidden layers, output (so we can index into it)
        self.layer_sizes=[no_input]+neurons_in_each_layer+[no_output]
        self.weights=[]
        self.biases=[]
        self.epsilon=1 #not used anymore: exploration now comes from mutation, not random actions
        for i in range(self.total_layers-1):
            self.biases.append(np.random.randn(self.layer_sizes[i+1], 1)*0.01)

    def initialise_weights(self):
        for i in range(self.total_layers-1):
            #every weight matrix is (neurons going out) x (neurons coming in)
            self.weights.append(np.random.randn(self.layer_sizes[i+1], self.layer_sizes[i]))

    def activation(self, x):
            #reLu 
            return np.maximum(0,x)
    
    def calculation(self, state):
        #we pass the inner calculations through an activiation function before passing them to the next layer
        #tbe state is the input (a column vector, shape no_input x 1)
        activations=[state]
        z_values=[] #the summation results
        a=state
        for i in range(self.total_layers-1):
            if i==0:    
                result=self.weights[0]@state + self.biases[0]
                a=self.activation(result)
                z_values.append(result)
                activations.append(a)
            elif i==self.total_layers-2:
                result=self.weights[-1]@a + self.biases[-1]
                z_values.append(result)
                #the two actions, each squashed on its own because they are independent
                #turn: tanh gives -1 (hard left) to 1 (hard right)
                turn=np.tanh(result[0,0])
                #gulp: sigmoid gives 0 to 1 (treat > 0.5 as "gulping")
                gulp=1/(1+np.exp(-np.clip(result[1,0], -500, 500)))
                result=np.array([[turn],[gulp]])
                return result, activations, z_values
            else:
                result=self.weights[i]@a + self.biases[i]
                a=self.activation(result)
                z_values.append(result)
                activations.append(a)

    def action(self, current_state):
        #no argmax and no random exploring here: the network outputs both actions directly
        output, _, _=self.calculation(current_state)
        return float(output[0,0]), float(output[1,0]) #(turn, gulp)

    #---------------- genetic algorithm helpers ----------------
    #the genome is every weight and bias flattened into one long list of numbers

    def get_genome(self):
        return np.concatenate([w.flatten() for w in self.weights] + [b.flatten() for b in self.biases])

    def set_genome(self, genome):
        idx=0
        for i in range(len(self.weights)):
            size=self.weights[i].size
            self.weights[i]=genome[idx:idx+size].reshape(self.weights[i].shape).copy()
            idx+=size
        for i in range(len(self.biases)):
            size=self.biases[i].size
            self.biases[i]=genome[idx:idx+size].reshape(self.biases[i].shape).copy()
            idx+=size

    def make_child(self, genome):
        #a brand new brain with the same shape, loaded with the given genome
        child=feed_forward_network(self.layers, self.no_input, self.no_output, self.neurons_in_each_layer)
        child.initialise_weights()
        child.set_genome(genome)
        return child


#---------------- genetic algorithm ----------------

def crossover(genome_a, genome_b):
    #each gene is copied from one parent or the other at random
    mask=np.random.rand(len(genome_a))<0.5
    return np.where(mask, genome_a, genome_b)

def mutate(genome, mutation_rate=0.1, mutation_strength=0.3):
    #each gene has a small chance of getting a little bit of gaussian noise added
    mask=np.random.rand(len(genome))<mutation_rate
    return genome + mask*np.random.randn(len(genome))*mutation_strength

def evolve(population, fitnesses, survivor_fraction=0.2, elites=2, mutation_rate=0.1, mutation_strength=0.3):
    #population: list of feed_forward_network, fitnesses: one number per slime (same order)
    n=len(population)
    order=np.argsort(fitnesses)[::-1] #best first
    n_survivors=max(2, int(n*survivor_fraction))
    survivors=[population[i] for i in order[:n_survivors]]

    #elitism: the very best brains go through completely unchanged so we never lose progress
    new_population=[population[i].make_child(population[i].get_genome()) for i in order[:elites]]

    #everyone else is a mutated child of two random survivors
    while len(new_population)<n:
        parent_a, parent_b=random.sample(survivors, 2)
        child_genome=crossover(parent_a.get_genome(), parent_b.get_genome())
        child_genome=mutate(child_genome, mutation_rate, mutation_strength)
        new_population.append(parent_a.make_child(child_genome))
    return new_population
