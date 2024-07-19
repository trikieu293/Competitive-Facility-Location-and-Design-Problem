import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import random
import math
import itertools
import matplotlib.pyplot as plt

def exact(n_customer, beta, lamda, theta, seed):
    random.seed(seed)
    MAP_SIZE = 100
    CUSTOMERS = n_customer
    ATTRACTIVENESS_ATTRIBUTES = 2

    POTENTIAL_LOCATION = CUSTOMERS // 3
    EXISTING_COMPETITIVE_FACILITIES = POTENTIAL_LOCATION // 3
    AVAILABLE_LOCATIONS = POTENTIAL_LOCATION - EXISTING_COMPETITIVE_FACILITIES

    BETA = beta              # the distance sensitivity parameter
    LAMBDA = lamda           # the elasticity parameter
    THETA = theta            # sensitivity parameter of the utility function

    N = [node for node in range(1, CUSTOMERS + 1)]          # index of customers
    P = random.sample(N, POTENTIAL_LOCATION)                # index of potential locations
    C = random.sample(P, EXISTING_COMPETITIVE_FACILITIES)   # index of competitive facility locations
    S = [facility for facility in P if facility not in C]   # index of controlled facilities

    # initiating locations for nodes
    locations = {}
    for i in N:
        x = random.randint(1, MAP_SIZE - 1)
        y = random.randint(1, MAP_SIZE - 1)

        while (x, y) in locations.values():
            x = random.randint(1, MAP_SIZE - 1)
            y = random.randint(1, MAP_SIZE - 1)
        locations.update({i: (x, y)})

    # calculating distances between nodes
    distances = {}
    for i in N:
        for j in N:
            if i == j:
                distances.update({(i, j): 0.1})
            else:
                x1, y1 = locations.get(i)
                x2, y2 = locations.get(j)
                distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                distances.update({(i, j): round(distance, 2)})

    # creating dict of scenarios
    scenarios_attr = {}
    for k in range(1, ATTRACTIVENESS_ATTRIBUTES + 1):
        value = 2
        scenarios_attr.update({k: [level for level in range(value)]})

    product = itertools.product(*[k for k in scenarios_attr.values()])
    R = {}
    count = 1
    for item in product:
        R.update({count: list(item)})
        count += 1

    # creating dict of nodes' weight
    w = {}
    for i in N:
        w.update({i: random.randint(1, 5)})
    # initiating attractiveness for competitive facilities
    c_attractiveness = {}
    for c in C:
        c_attractiveness.update({c: random.randint(3, 5)})

    ### Help functions
    def get_attractiveness(scenario):
        # return 1 + 1 * sum(R.get(scenario))
        attractiveness = 1
        for s in R.get(scenario):
            attractiveness = attractiveness*((1 + s)**THETA)
        return attractiveness
    def get_cost(scenario):
        return 1 + 1 * sum(R.get(scenario))

    def get_utility(customer, facility, scenario):
        return get_attractiveness(scenario) * ((distances.get((customer, facility)) + 1) ** (-BETA))

    def get_u_c(customer):
        utility_sum = 0
        for c in C:
            utility_sum += c_attractiveness.get(c) * ((distances.get((customer, c)) + 1) ** (-BETA))
        return utility_sum
    def get_max_u_s(customer):
        utility_sum = 0
        for e in S:
            utility_sum += get_utility(customer, e, max(R.keys()))
        return utility_sum

    def get_g(utility):
        if utility == 0:
            return 0
        return 1 - math.exp(-LAMBDA * utility)

    def plot_map():
        fig, ax = plt.subplots()
        customer_nodes = [node for node in N if node not in P]
        for node in N:
            x_temp, y_temp = locations.get(node)
            if node in customer_nodes:
                ax.scatter(x_temp, y_temp, s=3 ** w.get(node), c="gray", alpha=0.7)
                ax.annotate(str(node), xy=(x_temp, y_temp), color="white", fontsize=w.get(node),
                            horizontalalignment='center', verticalalignment='center')
            if node in x_result["j"].unique():
                ax.scatter(x_temp, y_temp, s=2.5 ** (x_result.loc[x_result.j == node, "attractiveness"].values[0] + 3),
                        c="forestgreen", alpha=0.7)
                ax.annotate(str(node), xy=(x_temp, y_temp), color="white",
                            fontsize=x_result.loc[x_result.j == node, "attractiveness"].values[0],
                            horizontalalignment='center', verticalalignment='center')
            if node in C:
                ax.scatter(x_temp, y_temp, s=3 ** c_attractiveness.get(node), c="red", alpha=0.7)
                ax.annotate(str(node), xy=(x_temp, y_temp), color="white", fontsize=c_attractiveness.get(node),
                            horizontalalignment='center', verticalalignment='center')
        plt.show()
    
    ### Model
    model = gp.Model()
    x_index = [(j, r) for j in S for r in R.keys()]

    # Decision varibales
    x = model.addVars(x_index, vtype=GRB.INTEGER, name='x')
    u1 = model.addVars([i for i in N], vtype=GRB.CONTINUOUS, name='u1')
    u2 = model.addVars([i for i in N], vtype=GRB.CONTINUOUS, name='u2')
    u3 = model.addVars([i for i in N], vtype=GRB.CONTINUOUS, name='u3')
    u4 = model.addVars([i for i in N], vtype=GRB.CONTINUOUS, name='u4')

    for i in N:
        u1[i].LB = get_u_c(i)
        u1[i].UB = get_u_c(i) + get_max_u_s(i)
    # Objective Function
    model.setObjective(sum(w[i] * (1 - u4[i]) * u3[i] for i in N), GRB.MAXIMIZE)

    # Constraints
    for j in S:
        model.addConstr(sum(x[j, r] for r in R.keys()) <= 1, name="Constraints 1")

    model.addConstr(sum(sum(get_cost(r) * x[j, r] for r in R.keys()) for j in S) <= 5, name="Constraints 2")

    for i in N:
        model.addConstr(u1[i] == get_u_c(i) + sum(x[j, r] * get_utility(i, j, r) for j in S for r in R.keys()), name="ConstrU1")
        model.addConstr(u1[i] * u2[i] == 1.0, name="ConstrU2")
        model.addConstr(u3[i] == 1.0 - (get_u_c(i) * u2[i]), name="ConstrU3")
        e_lambda = math.exp(-LAMBDA)
        model.addGenConstrExpA(u1[i], u4[i], e_lambda, name="ConstrU4")

    model.Params.TimeLimit = 60*60
    # model.params.NonConvex = 2
    model.update()
    model.optimize()

    ### checking result
    x_result = pd.DataFrame(x.keys(), columns=["j", "r"])
    x_result["value"] = model.getAttr("X", x).values()
    x_result["cost"] = [get_cost(r) for r in x_result["r"]]
    x_result.drop(x_result[x_result.value < 0.9].index, inplace=True)
    x_result["attractiveness"] = [get_attractiveness(r) for r in x_result["r"]]

    plot_map()
        
    return [x_result, round(model.Runtime, 2), len(x_result.index)]

print(exact(50, 1, 0.0000001, 1, 103093))