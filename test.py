import sys
from sokoban import Sokoban
'''
# XSokoban files
filenames = []
for i in range(1,91):
    filenames.append('screens/screen.' + str(i))
lengths = [99, 147, 146, 357, 150, 110, 127, 256, 239, 561, 295, 236, 262, 366, 199, 319, 221, 189, 323, 532, 162, 472, 496, 591, 412, 213, 367, 324, 1000, 507, 307, 210, 229, 195, 421, 535, 435, 101, 731, 349, 278, 328, 156, 199, 312, 278, 213, 238, 174, 422, 132, 627, 215, 248, 140, 230, 244, 237, 283, 171, 275, 255, 459, 403, 221, 453, 417, 362, 517, 360, 340, 348, 454, 250, 386, 282, 490, 142, 177, 244, 197, 161, 202, 173, 381, 156, 253, 453, 588, 502]

'''
# 6 Test Files
filenames = ['puzzles/easy2.txt', 'puzzles/easy4.txt',
             'puzzles/mod1.txt', 'puzzles/mod3.txt',
             'puzzles/hard1.txt', 'puzzles/hard2.txt']
lengths = [9,21,33,23,176,156]

trials = 1

def run_search(filename, sol_length, mut_rate, on_goal_reward, pop_size, output_file):
    s = Sokoban(filename)
    b = s.new_board(filename)
    s.do_searches(sol_length, mut_rate, on_goal_reward, pop_size, output_file)

def run_searches(multiplier, mut_rate, on_goal_reward, pop_size, output_file):
    for i in range(len(filenames)):
        fd = open(output_file, 'a')
        fd.write(str(filenames[i]) + '\n')
        fd.close()
        sol_length = lengths[i] * multiplier
        for j in range(trials):
            run_search(filenames[i], sol_length, mut_rate, on_goal_reward, pop_size, output_file)
        fd = open(output_file, 'a')
        fd.write('\n')
        fd.close()

# Uncomment the tests you'd like to run.
'''
# XSokoban testing set
#run_searches(10, 1, 2, 100, "xsokoban.txt")

# Run with default values
run_searches(5,0.75,0,100,"base.txt")

# Run with sequnces lengths of 2x,10x,20x the known solution length.
run_searches(2,0.75,0,100,"length2x.txt")
run_searches(10,0.75,0,100,"length10x.txt")
run_searches(10,0.75,0,100,"length20x.txt")

# Run with 25%, 50%, 100% mutation chance.
run_searches(5,0.25,0,100,"mut25.txt")
run_searches(5,0.5,0,100,"mut50.txt")
run_searches(5,1,0,100,"mut100.txt")

# Run with box on goal rewards of 1,2,5,10,20
run_searches(5,0.75,1,100,"goal+1.txt")
run_searches(5,0.75,2,100,"goal+2.txt")
run_searches(5,0.75,5,100,"goal+5.txt")
run_searches(5,0.75,10,100,"goal+10.txt")
run_searches(5,0.75,20,100,"goal+20.txt")

# Run with population sizes of 10,50,500,1000
run_searches(5,0.75,0,10,"pop10.txt")
run_searches(5,0.75,0,50,"pop50.txt")
run_searches(5,0.75,0,500,"pop500.txt")
run_searches(5,0.75,0,1000,"pop1000.txt")
'''
