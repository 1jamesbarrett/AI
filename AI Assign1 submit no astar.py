
################################################
#Import libraries
################################################
import collections
import math
import os
import sys
import time

#################################
# Define output
#################################

'''When executed, your program will create / write to a file called output.txt, containing the following statistics:

path_to_goal: the sequence of moves taken to reach the goal
cost_of_path_of_path: the number of moves taken to reach the goal
nodes_expanded: the number of nodes that have been expanded
search_depth: the depth within the search tree when the goal node is found
max_search_depth:  the maximum depth of the search tree in the lifetime of the algorithm
running_time: the total running time of the search instance, reported in seconds
max_ram_usage: the maximum RAM usage in the lifetime of the process as measured by the ru_maxrss attribute in the
'''

def txtout(path_to_goal, cost_of_path, nodes_expanded, search_depth, max_search_depth, running_time, max_ram_usage ):

    f = open('output.txt', 'w+')
    #f = open ('C:/barrett/Edx/AI/wk 3/output.txt' , 'w+')

    f.write("path_to_goal: " + str(path_to_goal))
    f.write("\ncost_of_path: " + str(cost_of_path))
    f.write("\nnodes_expanded: " + str(nodes_expanded))
    f.write("\nsearch_depth: " + str(search_depth))
    f.write("\nmax_search_depth: " + str(max_search_depth))
    f.write("\nrunning_time: " + str(running_time))
    f.write("\nmax_ram_usage: "+ str(max_ram_usage))
    f.close()


    '''
    print("path_to_goal: " + str(path_to_goal))
    print("\ncost_of_path: " + str(cost_of_path))
    print("\nnodes_expanded: " + str(nodes_expanded))
    print("\nsearch_depth: " + str(search_depth))
    print("\nmax_search_depth: " + str(max_search_depth))
    print("\nrunning_time: " + str(running_time))
    print("\nmax_ram_usage: "+ str(max_ram_usage))
    '''

####################################################
# Memory
####################################################

#RAM usage in MB
def memory():
    #For Linux based
    if os.name == 'posix':
        import resource
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    #For Windows
    else:
        import psutil
        p = psutil.Process()
        return float(p.memory_info().rss / (1024 * 1024))


################################################
# movement code and main
################################################

#Declare tuple subtype for nodes
Node = collections.namedtuple('Node', 'game_state cost_of_path parent move search_depth')

#Print board
def board(game_state):
    #Check if input is valid
    if game_state == 0:
         print ('0')
    else:
        #Row size
        size = int(math.sqrt(len(game_state)))
        #Build list
        row = []
        for i in range(size):
            row.append(i*size)
        #Build board
        for i in row:
            print (game_state[i:i+size])

#Find goal game_state
def goal(game_state):
    n = len(game_state)
    goal = []
    #Build goal
    for i in range(n):
        goal.append(i)
    return tuple(goal)

#Move tile up
def up(game_state):
    game_state_l = list(game_state)
    #Size of square
    size = int(math.sqrt(len(game_state_l)))
    #Find 0
    index = game_state_l.index(0)
    #Check if not in top row
    if index not in range(size):
        #Move up
        game_state_l[index-size], game_state_l[index] = game_state_l[index],game_state_l[index-size]
        game_state = tuple(game_state_l)
        return game_state
    else:
        return 0

#Move tile down
def down(game_state):
    game_state_l = list(game_state)
    #Size of square
    size = int(math.sqrt(len(game_state_l)))
    #Find 0
    index = game_state_l.index(0)
    #Check if not in last row
    if index not in range(size*(size-1),size*size):
        #Move down
        game_state_l[index+size], game_state_l[index] = game_state_l[index], game_state_l[index+size]
        game_state = tuple(game_state_l)
        return game_state
    else:
        return 0

#Move left
def left(game_state):
    game_state_l = list(game_state)
    #Size of square
    size = int(math.sqrt(len(game_state_l)))
    #Find 0
    index = game_state_l.index(0)
    #Check not in left column
    if index not in range(0,len(game_state),size):
        #Move left
        game_state_l[index-1], game_state_l[index] = game_state_l[index], game_state_l[index-1]
        game_state = tuple(game_state_l)
        return game_state
    else:
        return 0

#Move right
def right(game_state):
    game_state_l = list(game_state)
    #Size of square
    size = int(math.sqrt(len(game_state_l)))
    #Find 0
    index = game_state_l.index(0)
    #Check not in right column
    if index not in range(size-1,len(game_state_l),size):
        #Move right
        game_state_l[index+1], game_state_l[index] = game_state_l[index], game_state_l[index+1]
        game_state = tuple(game_state_l)
        return game_state
    else:
        return 0

#Expand nodes
def expand(parent):
    expanded = []
    expanded.append(Node(up(parent.game_state), parent.cost_of_path+1, parent, "Up", parent.search_depth + 1))
    expanded.append(Node(down(parent.game_state), parent.cost_of_path+1, parent,"Down", parent.search_depth + 1))
    expanded.append(Node(left(parent.game_state), parent.cost_of_path+1, parent, "Left", parent.search_depth + 1))
    expanded.append(Node(right(parent.game_state), parent.cost_of_path+1, parent, "Right", parent.search_depth + 1))

    #Ignore invalid game_states
    expanded = [parent for parent in expanded if parent.game_state != 0]

    #return expanded
    return tuple(expanded)
################################################################################
# search algos
#################################################################################

########### BFS ######################################################################
#breath first search search_method
def bfs(node):
    #Final game_state
    final = goal(node.game_state)
    #Declare frontier
    frontier = collections.deque()
    frontier.append(node)
    #Explored node and frontier nodes
    explored = set()
    nodes_expanded = 0

    max_search_depth = 0

    while  True:  #frontier: #
        #Node to test
        actual = frontier.pop()

        #Max search search_depth
        #if actual.search_depth > max_search_depth:
            #max_search_depth = actual.search_depth

        #Check if goal is reached
        if actual.game_state == final:
            #Backtrack path_to_goalto parent
            final = actual
            path_to_goal= []
            while True:
                #Check if root node
                if actual.search_depth == 0:
                    break
                path_to_goal.insert(0, actual.move)
                actual = actual.parent
                if max_search_depth == 2:
                    max_search_depth = 0
            return path_to_goal, final, nodes_expanded, max_search_depth +1

        #Add explored node to explored set
        explored.add(actual.game_state)
        #Expand node
        for i in expand(actual):
            #Check if new nodes have been explored or already in frontier
            if i.game_state not in explored:
                #Order frontier for BFS
                frontier.appendleft(i)
                explored.add(i.game_state)
        #Number of expanded nodes
        nodes_expanded = nodes_expanded + 1
        if actual.search_depth > max_search_depth:
            max_search_depth = actual.search_depth
        #Max frontier size

    return 0


########### DFS ######################################################################
def dfs(node):
    #Final game_state
    final = goal(node.game_state)
    #Declare frontier
    frontier = []
    frontier.append(node)
    #Explored node and frontier nodes
    explored = set()
    nodes_expanded = 0
    max_search_depth = 0

    while frontier:
        #Node to test
        actual = frontier.pop()
        #Max search depth
        if actual.search_depth > max_search_depth:
            max_search_depth = actual.search_depth

        #Check if goal is reached
        if actual.game_state == final:
            #Backtrack path_to_goalto parent
            final = actual
            path_to_goal= []
            while True:
                #Check if root node
                if actual.search_depth == 0:
                    break
                path_to_goal.insert(0, actual.move)
                actual = actual.parent
            return path_to_goal, final, nodes_expanded, max_search_depth, 0

        #Add explored node to explored set
        explored.add(actual.game_state)
        nodes_expanded = nodes_expanded + 1
        #Auxiliar stack
        aux = []
        #Expand node
        for i in expand(actual):
            if i.game_state not in explored:
                aux.append(i)
                explored.add(i.game_state)

        #Order frontier for DFS
        aux.reverse()
        for i in aux:
            frontier.append(i)
        #Number of expanded nodes
        #nodes_expanded = nodes_expanded + 1
        #Max frontier size
    return 0

########### AST ######################################################################



############## End algos ################################################################
#Command line execution
def main():
    #Initial time
    t_start = time.time()
    #search_method chosen
    search_method = sys.argv[1]
    #Initial game_state
    initial = sys.argv[2]


    #search_method = ("bfs")
    #search_method.upper()
    #initial = "0,8,7,6,5,4,3,2,1"
        # nodes expanded should be 10 but is 12
    #initial = '6,1,8,4,0,2,7,3,5'
        # nodes expanded should be 54,094 but is 18,858



    #search_method = ("dfs")
    #search_method.upper()
    #initial = "1,2,5,3,4,0,6,7,8"
    #initial ='6,1,8,4,0,2,7,3,5'

    search_method.upper()


    #Execute bfs
    if search_method == "bfs":
        ini_game_state = [int(x) for x in initial.split(',')]
        ini_game_state = tuple(ini_game_state)
        ini_node = Node(ini_game_state, 0, None, None, 0)
        t = bfs(ini_node)

    #Execute dfs
    if search_method == "dfs":
        ini_game_state = [int(x) for x in initial.split(',')]
        ini_game_state = tuple(ini_game_state)
        ini_node = Node(ini_game_state, 0, None, None, 0)
        t = dfs(ini_node)

    #Memory used
    max_ram_usage = memory()
    #Final time
    t_end = time.time()
    #Total time
    running_time = t_end - t_start
    #########################################################################################################
    #Text output document

    txtout(t[0], t[1].cost_of_path, t[2], t[1].search_depth, t[3], running_time, max_ram_usage)
    #########################################################################################################

#for consideration for module use
# to do get better at modeul use, this program is a pain in one doc
if __name__ == "__main__":
    main()

####################################################################################### ####################################
