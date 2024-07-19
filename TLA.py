import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import itertools
import time

def tla(n_customer, alpha, beta, lamda, theta, seed):
    random.seed(seed)
    MAP_SIZE = 100
    CUSTOMERS = n_customer
    ATTRACTIVENESS_ATTRIBUTES = 2

    POTENTIAL_LOCATION = CUSTOMERS // 3
    EXISTING_COMPETITIVE_FACILITIES = POTENTIAL_LOCATION // 3
    AVAILABLE_LOCATIONS = POTENTIAL_LOCATION - EXISTING_COMPETITIVE_FACILITIES

    ALPHA = alpha            # approximation level
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
            attractiveness = attractiveness * ((1 + s) ** THETA)
        return attractiveness
    def get_cost(scenario):
        return 1 + 1 * sum(R.get(scenario))

    def get_utility(customer, facility, scenario):
        return get_attractiveness(scenario) * ((distances.get((customer, facility)) + 1) ** (-BETA))

    def get_g(utility):
        if utility == 0:
            return 0
        return 1 - math.exp(-LAMBDA * utility)

    def get_g_derivative(utility):
        if utility == 0:
            return 0
        return LAMBDA * math.exp(-LAMBDA * utility)

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

    def get_interval_limit(customer):
        return get_max_u_s(customer)

    def get_omega(utility, customer):
        return get_g(utility + get_u_c(customer)) * (1 - (get_u_c(customer) / (utility + get_u_c(customer))))

    def get_omega_derivative(utility, customer):
        return get_g_derivative(utility + get_u_c(customer)) * (1 - (get_u_c(customer) / (utility + get_u_c(customer)))) + get_g(utility + get_u_c(customer)) * ((get_u_c(customer) / ((utility + get_u_c(customer)) ** 2)))

    def get_l(utility, customer, c_t):
        return get_omega(c_t, customer) + get_omega_derivative(c_t, customer) * (utility - c_t)

    def is_same_sign(a, b):
        return a * b > 0

    def diff_function_25(utility, customer, point):
        return get_l(utility, customer, point) - get_omega(utility, customer) * (1.0 + ALPHA)


    def diff_function_24(utility, customer, c):
        return get_omega_derivative(utility, customer) * (utility - c) + get_omega(c, customer) - get_omega(utility, customer)

    def bisect(func, low, high, customer, c):
        temp = high
        midpoint = (low + high) / 2.0
        while (high - low)/2 >= 0.00000001:
            midpoint = (low + high) / 2.0
            if is_same_sign(func(low, customer, c), func(midpoint, customer, c)):
                low = midpoint
            else:
                high = midpoint
        # if midpoint >= (1 - 0.0000000001) * temp:
        #     return temp
        return midpoint

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
    
    ### The TLA procedure
    l_dict = {}
    a_dict = {}
    b_dict = {}
    c_dict = {}
    
    start_time_tla = time.time()
    for customer in N:
        def tla():
            # Step 1
            l = 1
            c = 0
            c_dict.update({(customer, l): 0})
            c_t = 0
            b_dict.update({(customer, 1): get_omega_derivative(0, customer)})
            phi_bar = get_interval_limit(customer)

            # Step 2
            while get_l(phi_bar, customer, c_t) >= get_omega(phi_bar, customer) * (1.0 + ALPHA):
                root = bisect(diff_function_25, c, phi_bar, customer, c_t)
                c_dict.update({(customer, l + 1): root})
                if root == phi_bar:
                    l_dict.update({customer: l})
                    break
                else:
                    c = c_dict.get((customer, l + 1))
                    l = l + 1
                    if get_omega(phi_bar, customer) >= (get_omega_derivative(phi_bar, customer) * (phi_bar - c) + get_omega(c, customer)):  # (23) hold -> Step 3b
                        c_t = bisect(diff_function_24, c, phi_bar, customer, c)
                        b_dict.update({(customer, l): get_omega_derivative(c_t, customer)})
                    else: # (23) not hold -> Step 3a
                        l_dict.update({customer: l})
                        c_dict.update({(customer, l + 1): phi_bar})
                        if get_omega(c_dict.get((customer, l)), customer) * (1.0 + ALPHA) <= get_omega(phi_bar, customer):
                            value = (get_omega(phi_bar, customer) - get_omega(c, customer) * (1.0 + ALPHA)) / (phi_bar - c)
                            b_dict.update({(customer, l): value})
                        else:
                            b_dict.update({(customer, l): 0})
                            break
                    if c_t == phi_bar: # Step 4
                        c_dict.update({(customer, l + 1): c_t})
                        l_dict.update({customer: l})
                        break

            if get_l(phi_bar, customer, c_t) < get_omega(phi_bar, customer) * (1 + ALPHA):
                c_dict.update({(customer, l + 1): phi_bar})
                l_dict.update({customer: l})

        tla()

    for customer in N:
        check = l_dict.get(customer)
        for l in range(1, l_dict.get(customer) + 1):
            a_dict.update({(customer, l): c_dict.get((customer, l + 1)) - c_dict.get((customer, l))})
    end_time_tla = time.time()
    time_tla = round(end_time_tla - start_time_tla, 2)
    
    ### Model
    model = gp.Model()
    x_index = [(j, r) for j in S for r in R.keys()]

    # Decision varibales
    x = model.addVars(x_index, vtype=GRB.INTEGER, name='x')
    y = model.addVars([(i, l) for (i, l) in b_dict.keys()], vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name='y')

    # Objective Function
    model.setObjective(sum(sum(w[i] * a_dict.get((i, l)) * b_dict.get((i, l)) * y[i, l] for l in range(1, l_dict.get(i) + 1)) for i in N), GRB.MAXIMIZE)

    # Constraints
    for i in N:
        model.addConstr(sum(get_utility(i, j, r) * x[j, r] for j in S for r in R.keys()) == sum(a_dict.get((i, l)) * y[i, l] for l in range(1, l_dict.get(i) + 1)), name="Constraints 1")

    for j in S:
        model.addConstr(sum(x[j, r] for r in R.keys()) <= 1, name="Constraints 2")

    model.addConstr(sum(sum(get_cost(r) * x[j, r] for r in R.keys()) for j in S) <= 5, name="Constraints 3")

    # for i in N:
    #     for l in range(1, l_dict.get(i) - 1):
    #         model.addConstr(y[i, l] >= y[i, l + 1], name="Constrt test")

    model.Params.TimeLimit = 60*60
    model.update()
    model.optimize()

    ### checking result
    x_result = pd.DataFrame(x.keys(), columns=["j", "r"])
    x_result["value"] = model.getAttr("X", x).values()
    x_result["cost"] = [get_cost(r) for r in x_result["r"]]
    x_result["attractiveness"] = [get_attractiveness(r) for r in x_result["r"]]
    x_result.drop(x_result[x_result.value < 0.9].index, inplace=True)

    y_result = pd.DataFrame(y.keys(), columns=["i", "l"])
    y_result["value"] = model.getAttr("X", y).values()

    plot_map()
    
    return [x_result, time_tla, round(model.Runtime, 2)]

print(tla(50, 0.05, 100, 1, 1, 103093))
