import sys
import argparse
import numpy as np

from copy import deepcopy

##############################################################
class CSP():
    def __init__(self):
        self.Constraints = dict()  #
    #########################################################################################
    def AC3(self, Grid_Loc_N_Val, Current_Spot_Val):#
        queue = set([])
        # initialise the queue
        for k in self.Constraints.keys():
            #print(k)
            #print('k in constraints')
            #(0,1) (1,0) (0,2) (2,0)....
            queue.add(k)

        while len(queue) > 0:
            Xi, Xj = queue.pop()
            if self.revise(Current_Spot_Val, Xi, Xj):
                if len(Current_Spot_Val[Xi]) == 0:
                    keep_count()
                    return False
                for Xk in self.neighbours(Xi, [Xj]):
                    queue.add((Xk, Xi))

        return True
    #########################################################################################
    # given a choice of x for Xi, are there any values in Dj that satisfy the constraints between Xi and Xj?
    def can_be_satisfied(self, Current_Spot_Val, x, Xi, Xj):
        satisfactory = False
        for d in Current_Spot_Val[Xj]:
            for c in self.Constraints[Xi, Xj]:
                #print (c)
                if c(x, d):
                    #print(c(x, d))
                    #print('c(x, d)')
                    satisfactory = True
        return satisfactory
    #########################################################################################
    def revise(self, Current_Spot_Val, Xi, Xj): #
        revised = False
        for x in Current_Spot_Val[Xi]:
            if not self.can_be_satisfied(Current_Spot_Val, x, Xi, Xj):
                Current_Spot_Val[Xi].remove(x)

                revised = True
        return revised
    #########################################################################################
    # find the set of neighbours excepting all members of Ex
    def neighbours(self, X, Ex):
        # you are a neighbour if you share a binary constraint...
        result = [n for x, n in self.Constraints.keys() if x == X and not n not in Ex]
        return result

#########################################################################################
def init_csp(grid):
    Constraints = CSP()
    indexes = []

    Grid_Loc_N_Val = dict()
    Current_Spot_Val = dict()  # Domain values
    tmp = [(i, int(d)) for i, d in enumerate(grid)]
    #print('tmp')
    #print(tmp)
    for i, x in tmp:
        Grid_Loc_N_Val[i] = x
        #print(Grid_Loc_N_Val)

    # This sets up the initial domains of each cell, in preparation for the intial assignments to be inserted
    for x in range(0, 81):
        dx = Grid_Loc_N_Val[x]
        #print(dx)
        #print('dx')
        #dc is value from specified location 0-80
        if dx != 0:
            Current_Spot_Val[x] = [dx]
        else:
            Current_Spot_Val[x] = [v for v in range(1, 10)]

    for row in range(0, 9):
        indexes = [x for x in range(row * 9, (row + 1) * 9)]
        # define value for each position location in grid
        #print(row)
        #print(indexes)
        #print('index done')
        setup_all_diff(Constraints, indexes)

    for col in range(0, 9):
        indexes = []
        for row in range(0, 9):
            indexes.append((row * 9) + col)
        setup_all_diff(Constraints, indexes)

    #############################
    indexes = []
    for row in range(0, 3):
        for col in range(0, 3):
            indexes.append((row * 9) + col)
            #print(indexes)
            #print('index append done')
            #[0, 1, 2, 9, 10, 11, 18, 19, 20]
    setup_all_diff(Constraints, indexes)

    indexes = []
    for row in range(3, 6):
        for col in range(0, 3):
            indexes.append((row * 9) + col)
    setup_all_diff(Constraints, indexes)

    indexes = []
    for row in range(6, 9):
        for col in range(0, 3):
            indexes.append((row * 9) + col)
    setup_all_diff(Constraints, indexes)

    #############################
    indexes = []
    for row in range(0, 3):
        for col in range(3, 6):
            indexes.append((row * 9) + col)
    setup_all_diff(Constraints, indexes)

    indexes = []
    for row in range(3, 6):
        for col in range(3, 6):
            indexes.append((row * 9) + col)
    setup_all_diff(Constraints, indexes)

    indexes = []
    for row in range(6, 9):
        for col in range(3, 6):
            indexes.append((row * 9) + col)
    setup_all_diff(Constraints, indexes)

    #############################
    indexes = []
    for row in range(0, 3):
        for col in range(6, 9):
            indexes.append((row * 9) + col)
    setup_all_diff(Constraints, indexes)

    indexes = []
    for row in range(3, 6):
        for col in range(6, 9):
            indexes.append((row * 9) + col)
    setup_all_diff(Constraints, indexes)

    indexes = []
    for row in range(6, 9):
        for col in range(6, 9):
            indexes.append((row * 9) + col)
    setup_all_diff(Constraints, indexes)

    return Constraints, Grid_Loc_N_Val, Current_Spot_Val

#####################################################################################
def setup_all_diff(csp: CSP, xs): #
    for i in range(0, len(xs)):
        for j in range(i + 1, len(xs)):
            pair = (xs[i], xs[j])
            pair2 = (xs[j], xs[i])
            if pair not in csp.Constraints:
                csp.Constraints[pair] = []
            csp.Constraints[pair].append(lambda x, d: x != d)
            if pair2 not in csp.Constraints :  ######check all ajacent squares
                csp.Constraints[pair2] = []
            csp.Constraints[pair2].append(lambda x, d: x != d)

#######################################################################################
def backtracking_search(Constraints, Grid_Loc_N_Val, Current_Spot_Val) : # , Method_Used = ' BTS'):
    # keep_count()   # wrong spot
    return backtrack(Constraints, Grid_Loc_N_Val, Current_Spot_Val) #, Method_Used)

################################################################
# return [] to signify failure
def inference(csp, var, value):
    return []

#################################################################
def complete(Grid_Loc_N_Val):
    for k in Grid_Loc_N_Val.keys():
        if Grid_Loc_N_Val[k] == 0:
            return False
    return True

#################################################################
def get_next_unassigned_position(Grid_Loc_N_Val):
    for i in range(0, 81):
        if Grid_Loc_N_Val[i] == 0:
            return i

############################################################################################
def backtrack(Constraints: CSP, Grid_Loc_N_Val: dict, Current_Spot_Val: dict): #, Method_Used = ' BTS'):
    if complete(Grid_Loc_N_Val):
        return Grid_Loc_N_Val
    i = get_next_unassigned_position(Grid_Loc_N_Val)

    # for each possible assignment to that position
    for d in Current_Spot_Val[i]:
        # make a copy of the board and update it with the assignment
        Grid_Loc_N_Val_ = deepcopy(Grid_Loc_N_Val)
        Current_Spot_Val_ = deepcopy(Current_Spot_Val)
        Grid_Loc_N_Val_[i] = d
        Current_Spot_Val_[i] = [d]

        # check if the board as assigned is consistent
        if Constraints.AC3(Grid_Loc_N_Val_, Current_Spot_Val_):
            # otherwise move on to make a new assignment in another position
            solution = backtrack(Constraints, Grid_Loc_N_Val_, Current_Spot_Val_)
            if solution is not None:
                #print (solution)
                return (solution)

    # if we got here (through every possible assignment of values) without finding an assignment that works, then this assignment must be rejected
    # we need to backtrack and try something else.
    return None
##########################################################################################################

def Grid_Loc_N_Val_2_str(Grid_Loc_N_Val):
    result = ''
    #print(Grid_Loc_N_Val)
    #print('gridd val fin')
    for x in range(0, 81):
        result += str(Grid_Loc_N_Val[x])
    return result
#################################################
def keep_count():
    keep_count.counter += 1
    return  keep_count.counter
keep_count.counter = 0

def Method_Track(tracker):
    Method_Tracked = ' AC3'
    if tracker > 3:
       Method_Tracked = ' BTS'
    return Method_Tracked
#################################################################

##################################################################################
def solve(problem):
    Constraints, Grid_Loc_N_Val, Current_Spot_Val = init_csp(problem)

    # parse problem string
    # build representation
    # for i, v in enumerate(grid):
    #     csp.Current_Spot_Val[i] = [v]
    # solve problem

    ok = Constraints.AC3(Grid_Loc_N_Val, Current_Spot_Val) #, Method_Used)
    #Method_used = ' BTS'
    if not ok:
        raise Exception('errm')
    solution = backtrack(Constraints, Grid_Loc_N_Val, Current_Spot_Val) #, Method_Used)
    return solution
##################################################################################
def main():
    #solution = solve(sys.argv[1])
    #with open('output.txt', 'w') as of:
    #with open('C:/barrett/Edx/AI/Assign 4/output.txt', 'w') as of:
    #    of.write(Grid_Loc_N_Val_2_str(solution))
#############################################################################
# single input and output
    #Method_Used = ' ref'
    #problem = '000000000302540000050301070000000004409006005023054790000000050700810000080060009'
    #problem  = '000260701680070090190004500820100040004602900050003028009300074040050036703018000'
    problem = sys.argv[1]
    #print(problem)
    #print(Method_Used)
    #check_val= solve(problem.strip())
    #print(check_val)
    actual = Grid_Loc_N_Val_2_str(solve(problem))
    #print(keep_count())
    MTH_Used = Method_Track(keep_count())
    actual = actual +  MTH_Used
    #print('should be solution')
    #print(actual.strip())
    #output_file = 'C:/barrett/Edx/AI/Assign 4/output.txt'
    output_file = 'output.txt'
    with open(output_file, 'w') as of:
        of.write(actual)


#########################################################################################################

if __name__ == '__main__':
    main()

##################################################
