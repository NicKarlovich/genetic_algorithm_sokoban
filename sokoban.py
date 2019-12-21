from board import Board
from spot import Spot
from direction import Direction
import bisect
import math
import random
import sys
from munkres import Munkres, print_matrix
import time
import copy

# Global Variables set by input parameters in do_search
POP_NUMBER = 50
SOL_LENGTH = 75
BALL_ON_GOAL_REWARD = 0

# Constance Time to solve, never changed for our tests.
TIME_TO_SOLVE = 300

class Sokoban:
    def __init__(self, filename):
        self.board = self.new_board(filename)
        distanceDict = self.make_distances_dict()
        self.distanceDict = distanceDict
        self.MAX_BOARD_VAL = self.compute_max_fitness()

    ''' Initial creation of dict of 2D arrays for BFS-Heuristic '''
    def make_distances_dict(self):
        #reads files and splits into lines
        fd = open(self.board.filename, 'r')
        grid = fd.read().split('\n')
        for i in range(len(grid)):
            grid[i] = list(grid[i])

        goals = []
        grid_lst = []
        #creates list of distances and goals to check
        for i in range(len(grid)):
            grid_lst.append([])
            for j in range(len(grid[i])):
                if grid[i][j] == "." or grid[i][j] == '*':
                    goals.append((j,i))
                grid_lst[i].append(float('inf'))

        distance_matrix_dict = {}
        for goal in goals:
            grid_lst_copy = copy.deepcopy(grid_lst)
            distance_dict = {}
            q = [goal]
            dist = 0
            while q != []:
                num = len(q)
                for i in range(num):
                    cur = q[0]
                    grid_lst_copy[cur[1]][cur[0]] = dist
                    del q[0]
                    #left, right, up, down
                    moves = [(cur[0]-1, cur[1]), (cur[0]+1, cur[1]), (cur[0], cur[1]-1), (cur[0], cur[1]+1)]
                    for move in moves:
                        y = move[1]
                        x = move[0]
                        if dist < grid_lst_copy[y][x] and grid[y][x] != "#":
                            q.append(move)
                dist += 1
            distance_matrix_dict[goal] = grid_lst_copy
        return distance_matrix_dict
        
    ''' Creates new board from file '''
    def new_board(self, filename):
        e = []  # empty solution list
        b = Board(e, filename)
        with open(filename, 'r') as f:  # automatically closes file
            read_data = f.read()
            lines = read_data.split('\n')
            height = lines.pop(0)
            x = 0
            y = 0
            for line in lines:
                for char in line:
                    # adds Spots to board's sets by reading in char
                    if char == '#':
                        b.add_wall(x, y)
                    elif char == '.':
                        b.add_goal(x, y)
                    elif char == '@':
                        b.set_player(x, y)
                    elif char == '+':
                        # player gets its own Spot marker
                        b.set_player(x, y)
                        b.add_goal(x, y)
                    elif char == '$':
                        b.add_box(x, y)
                    elif char == '*':
                        b.add_box(x, y)
                        b.add_goal(x, y)
                    x += 1
                y += 1
                x = 0
            # check for a board with no player
            if hasattr(b, 'player'):
                return b
            else:
                print ("No player on board")
                return None

    ''' Computes distance all boxes are from goals intially, used as max distance for heuristic calculation.  See 3.3 Scoring Method in Final Paper '''
    def compute_max_fitness(self):
        payoff_mtx = []
        lst_goals = self.board.goals
        lst_boxes = self.board.boxes

        # for each goal, get heuristic distance between boxes
        for goal in lst_goals:
            box_line = []
            for box in lst_boxes:
                # box_line.append(self.manhattan(goal.x, goal.y, box.x, box.y))
                box_line.append(self.bfs_dist(goal.x, goal.y, box.x, box.y))
            payoff_mtx.append(box_line)
            
        m = Munkres() # Do hungarian calculation.
        indexes = m.compute(payoff_mtx)
        total = 0
        # sum values obtained from Munkres into a total score
        for row, column in indexes:
            value = payoff_mtx[row][column]
            total += value
        return total

    ''' Helper function to convert between Sokoban game and genetic algorithm '''
    def convert_string_to_direction(self, action):
        if action == 'u':
            return Direction(Spot(0, -1), 'u')
        elif action == 'd':
            return Direction(Spot(0, 1), 'd')
        elif action == 'r':
            return Direction(Spot(1, 0), 'r')
        elif action == 'l':
            return Direction(Spot(-1, 0), 'l')
        else:
            print("Couldn't convert string to direction ERROR!")
            return None

    ''' similar to search() but returns boolean, less output aswell '''
    def winner(self, board, sequence):
        if board.is_win():
            print("found a winner")
            return True
        l_seq = len(sequence)
        #for each character in sequence
        for char_idx in range(l_seq):
            next_move = self.convert_string_to_direction(sequence[char_idx])
            if board.is_valid(next_move): #check if move valid, if it's not, then we can just ignore move.
                board.move(self.convert_string_to_direction(sequence[char_idx]))
            else:
                if board.is_win():
                    print("found a winner")
                    return True
        return False

    ''' 
    takes a board and sequences, run sequence through the inputted board
    Returns (board_obj, number_of_moves_made)
    '''
    def search(self, board, sequence, output = False):
        if output:
            print(sequence)
            print("search start")
            for x in board.boxes:
                print(str(x))
        
        if board.is_win():
            return (board, 0) # (the board, the number of moves it took)

        l_seq = len(sequence)
        #for each character in sequence
        for char_idx in range(l_seq):
            next_move = self.convert_string_to_direction(sequence[char_idx])
            if board.is_valid(next_move): #check if move valid, if it's not, then we can just ignore move.
                board.move(self.convert_string_to_direction(sequence[char_idx]))
            else:
                if output:
                    print("invalid move at idx " + str(char_idx))
                if board.is_win():
                    if output:
                        print('board sequence win!')
                        print('seq: ' + ("".join(sequence)))
                        print('idx: ' + str(char_idx))
                    return (board, char_idx) # (the board, the number of moves it took)
        if output:
            print("search end")
            for x in board.boxes:
                print(str(x))
        return (board, l_seq)

    '''
    creates a distance matrix as such, where each entry is the distance (via the respective fitness function) from a ball to a goal.
    
                balls
               b1  b2  b3  b4
    g         ---------------
    o    g1  |   |   |   |   |
    a    g2  |   |   |   |   |
    l    g3  |   |   |   |   |
    s    g4  |   |   |   |   |
    
    After computing and filling the distance matrix (between each ball and goal) we run the hungarian algorithm
    which will solve the combinatorial opitimzation problem.
    
    https://en.wikipedia.org/wiki/Hungarian_algorithm
    
    It will then return the distance matrix and the indices it found that were optimal which will later be used to compute the heuristic.
    '''
    def find_distances(self, goal_list, box_list, idx, payoff_mtx, output = False):
        #for every goal, get heuristic distance
        for goal in goal_list:
            box_line = []
            for box in box_list:
                # box_line.append(self.manhattan(goal.x, goal.y, box.x, box.y))
                box_line.append(self.bfs_dist(goal.x, goal.y, box.x, box.y))
                if output:
                    print("x: " + str(goal.x) + "  y: " + str(goal.y))
                    print("x: " + str(box.x) + "  y: " + str(box.y))
            payoff_mtx.append(box_line)

        #solve matrix
        m = Munkres()
        temp = m.compute(payoff_mtx)
        if output:
            print("-start-")
            print_matrix(temp)
            print("-end-")
        return (temp, payoff_mtx)

    ''' 
    Run at the beginning of a generation in genetic algorithm
    This function "precomputes" each gene by running each gene and then returning heuristic value
    This way when fitness() function is called, it's not actually calculating the fitness its just pulling the value from fitness_dict defined here.
    '''
    def compute_fitness(self, pop_arr, fitness_dict, output = False):
            total_fitness = 0
            board_arr = []
            for i in range(0,POP_NUMBER):
                board_arr.append(self.new_board(self.board.filename))
            for gene_idx in range(0,len(pop_arr)):
                if output:
                    print("-"*50)
                (board_arr[gene_idx], num_of_moves) = self.search(board_arr[gene_idx], pop_arr[gene_idx]) # do the search with this gene's moves.
                correct_counter = 0
                payoff_mtx = []
                # Does hungarian algorithm to find the cheapest manhattan distance to all goals from all balls.
                (indexes, payoff_mtx) = self.find_distances(board_arr[gene_idx].goals, board_arr[gene_idx].boxes, gene_idx, payoff_mtx)
                total = 0
                # SOL_LENGTH is the max number of moves made before the gene is out of moves.
                for row, column in indexes:
                    value = payoff_mtx[row][column] #minimum values of hungarian.
                    total += value
                    if value == 0: # a ball is on the goal state
                        correct_counter += 1
                    if output:
                        print("(" + str(row) + ", " + str(column) + ") -> " + str(value))
                if output:
                    print("total cost:  " + str(total))
                    print("gene: " + ''.join(pop_arr[gene_idx]))
                
                ''' Scoring for this heuristic function can be found explained in 3.3 Scoring Method in our paper '''
                # if the total hungarian distance of all balls from the goal is greater AND no balls are in goal states, then this "gene" is bad and should have heuristic 0.
                if(total > self.MAX_BOARD_VAL and correct_counter < 1):
                    total = 0 # worst score possible, the ball was moved further away from the goal state.
                else:
                    if output:
                        print("distance away: " + str(total))
                    total = SOL_LENGTH - num_of_moves - total + self.MAX_BOARD_VAL
                    if output:
                        print("score: " + str(total) +  " " + str(self.MAX_BOARD_VAL) )
                    total = total + (correct_counter * BALL_ON_GOAL_REWARD)
                    if output:
                       print("after: ball out output reward" + str(total))
                    if output:
                        print("total-score: " + str(total))
                        if BALL_ON_GOAL_REWARD > 0:
                            print("ball on goal!")
                fitness_dict[''.join(pop_arr[gene_idx])] = total
                total_fitness = total_fitness + total
            return total_fitness

    ''' Fitness calculation is done in compute_fitness, this function is basically a fancy getter '''
    def fitness_fn(self, fitness_dict, iteration=None):
        if iteration == None:
            return 0
        else:
            try:
                return fitness_dict[''.join(iteration)]
            except:
                print("\ninvalid string: " + (''.join(iteration)))
                print("printing fitness_dict")
                for x in fitness_dict:
                    print(str(x) + " " + str(fitness_dict[x]))
                return 0


    ''' Heuristic function Manhattan, ignores walls, just finds x-y distance to goal.
    Ex
    #####
    #   #
    # #o#
    #.# #
    #####
    The box is 'o', the goal is '.', the walls are '#'.  The manhattan score would be 2 (along x axis) + 1 (along y axis) = 3
    '''
    def manhattan(self, goal_x, goal_y, box_x, box_y):
            return abs(goal_x - box_x) + abs(goal_y - box_y)

    ''' Heuristic function BFS, ignores walls, just finds x-y distance to goal.
    The scores for this heuristic are computed once at the beginning in the form of a dictionary 
    of 2D arrays where each 2D array is keyed by the goal state's (x,y) position.
    
    Ex
    #####
    #   #
    # #o#
    #.# #
    #####
    The box is 'o', the goal is '.', the walls are '#'.  The manhattan score would be 1 (up) + 2 (left) + 2 (down) = 5
    '''
    def bfs_dist(self, goal_x, goal_y, box_x, box_y):
        test = self.distanceDict[(goal_x, goal_y+1)][box_y+1][box_x]
        return test
        
    ''' Runs the genetic algorithm, does recombination, mutation, fitness computation, exiting when solution is found '''
    def genetic_algorithm(self, population, fitness_dict, pmut=0.8):
        gene_pool=['u','d','l','r']
        start = time.time()
        new_best_score = ("",0)
        while time.time() - start < TIME_TO_SOLVE and not self.winner(self.new_board(self.board.filename), new_best_score[0]):
            lst_of_tuple_of_seq_fitness_percent = []

            # points are maximized and have nothing to do with number of moves.
            # Running of all genes through the puzzle happens here
            total_points = self.compute_fitness(population, fitness_dict)
            
            if total_points > 0:
                best_score = ("",0)

                for gene1 in population:
                    gene = "".join(gene1)
                    #this is score we're tyring to maximize
                    if fitness_dict[gene] > best_score[1]:
                        best_score = (gene, fitness_dict[gene])            
                    lst_of_tuple_of_seq_fitness_percent.append((gene, fitness_dict[gene], (fitness_dict[gene] / total_points)))
                
                a = sorted(lst_of_tuple_of_seq_fitness_percent, key=lambda tup: tup[2])
                new_best_score = best_score
                
                #keeping the best one from each generation of genes
                population[0] = best_score[0]

                for j in range(1, len(population)):
                    
                    # Creation of the Cummulative Probability Distribution
                    for i in range(len(lst_of_tuple_of_seq_fitness_percent)):
                        if i != 0:
                            lst_of_tuple_of_seq_fitness_percent[i] = (lst_of_tuple_of_seq_fitness_percent[i][0],
                            lst_of_tuple_of_seq_fitness_percent[i][1],
                            lst_of_tuple_of_seq_fitness_percent[i - 1][2] + lst_of_tuple_of_seq_fitness_percent[i][2])
                            # add the previous value to the next percentage

                    percentage = random.random()
                    
                    #default first values
                    permute1 = lst_of_tuple_of_seq_fitness_percent[0][0]
                    permute2 = lst_of_tuple_of_seq_fitness_percent[1][0]

                    # Deciding which gene should be permuted based on it's fitness score
                    for i in range(len(lst_of_tuple_of_seq_fitness_percent)):
                        if percentage < lst_of_tuple_of_seq_fitness_percent[i][2]:
                            permute1 = lst_of_tuple_of_seq_fitness_percent[i][0]
                            break
                        
                    percentage = random.random()
                    for i in range(len(lst_of_tuple_of_seq_fitness_percent)):
                        if percentage < lst_of_tuple_of_seq_fitness_percent[i][2]:
                            permute2 = lst_of_tuple_of_seq_fitness_percent[i][0]
                            break
                    
                    # Combine the two sequences
                    temp = self.soko_recombine(permute1, permute2)
                    # Mutate the recombined string if mutation rate is triggered.
                    population[j] = self.soko_mutate(list(temp), gene_pool, pmut)
            
            else:       #if the total score is 0 reset the population.
                population = (self.init_population(POP_NUMBER, ['u','d','l','r'], SOL_LENGTH))
        end = time.time()
        total = end - start
        self.compute_fitness(population, fitness_dict)
        
        m = 0
        best = population[0]
        
        for x in range(len(population)):
            m1 = (self.fitness_fn(fitness_dict, population[x]))
            if m1 >= m:
                m = m1
                best = population[x]
        return total, best

    def init_population(self, population_number, gene_pool, solution_length):
        '''
        population_number : Number of indiviuals in the population_number
        gene_pool         : A list of the options that the genetic algorithm can choose from.
        solution_length   : The number of moves we allow a test to take to get to a solution.
        '''
        g = len(gene_pool)
        population = []
        for i in range(population_number):
            new_individual = [gene_pool[random.randrange(0,g)] for j in range(solution_length)]
            population.append(new_individual)
        return population

    ''' Combines two strings together based on random index between '''
    def soko_recombine(self, x, y):
        n = len(x)
        if n == 0:
            n = SOL_LENGTH
        c = random.randrange(0, n)
        #print("Recombine: " + str(c) + "\n")
        return x[:c] + y[c:]

    ''' Mutates inputted string if probability is triggered '''
    def soko_mutate(self, x, gene_pool, mutate_rate):
            # no mutation occurs
            if random.uniform(0,1) >= mutate_rate:
                return x
            else:
                #mutation occurs
                n = len(x)
                if n == 0:
                    n = SOL_LENGTH
                g = len(gene_pool)
                c = random.randrange(0, n)
                r = random.randrange(0, g)
                new_gene = gene_pool[r]
                return x[:c] + [new_gene] + x[(c + 1):]

    ''' pseduo-main function '''
    def do_searches(self, sol_length, mut_rate, on_goal_reward, pop_size, output_file):
        global POP_NUMBER
        global SOL_LENGTH
        global BALL_ON_GOAL_REWARD
        POP_NUMBER = pop_size
        SOL_LENGTH = sol_length
        BALL_ON_GOAL_REWARD = on_goal_reward
        
        x = self.init_population(POP_NUMBER, ['u','d','l','r'], SOL_LENGTH)
        fitness_dict = {}
        
        o = self.genetic_algorithm(x, fitness_dict, mut_rate)
        fd = open(output_file, 'a')
        fd.write(str(fitness_dict[''.join(o[1])]))
        fd.write("\nTime: " + str(o[0]) + '\n')
        fd.close()
