import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import random
import math
import itertools
import time


def tla(n_customer, epsilon, beta, lamda, theta, seed):
    random.seed(seed)
    MAP_SIZE = 100
    CUSTOMERS = n_customer
    ATTRACTIVENESS_ATTRIBUTES = 2

    POTENTIAL_LOCATION = CUSTOMERS // 3
    EXISTING_COMPETITIVE_FACILITIES = POTENTIAL_LOCATION // 3
    AVAILABLE_LOCATIONS = POTENTIAL_LOCATION - EXISTING_COMPETITIVE_FACILITIES

    EPSILON = epsilon            # approximation level
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
        return get_l(utility, customer, point) - get_omega(utility, customer) * (1.0 + EPSILON)


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
            while get_l(phi_bar, customer, c_t) >= get_omega(phi_bar, customer) * (1.0 + EPSILON):
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
                        if get_omega(c_dict.get((customer, l)), customer) * (1.0 + EPSILON) <= get_omega(phi_bar, customer):
                            value = (get_omega(phi_bar, customer) - get_omega(c, customer) * (1.0 + EPSILON)) / (phi_bar - c)
                            b_dict.update({(customer, l): value})
                        else:
                            b_dict.update({(customer, l): 0})
                            break
                    if c_t == phi_bar: # Step 4
                        c_dict.update({(customer, l + 1): c_t})
                        l_dict.update({customer: l})
                        break

            if get_l(phi_bar, customer, c_t) < get_omega(phi_bar, customer) * (1 + EPSILON):
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

    model.addConstr(sum(sum(get_cost(r) * x[j, r] for r in R.keys()) for j in S) <= 9, name="Constraints 3")

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

    return [x_result, time_tla, round(model.Runtime, 2)]


def exact(n_customer, beta, lamda, theta,seed):
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

    model.addConstr(sum(sum(get_cost(r) * x[j, r] for r in R.keys()) for j in S) <= 9, name="Constraints 2")

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

    return [x_result, round(model.Runtime, 2), len(x_result.index)]

if __name__ == "__main__":
    def result(n_customer, epsilon, beta, lamda, theta, result_approximation, result_exact, seed):
        random.seed(seed)
        MAP_SIZE = 100
        CUSTOMERS = n_customer
        ATTRACTIVENESS_ATTRIBUTES = 2

        POTENTIAL_LOCATION = CUSTOMERS // 3
        EXISTING_COMPETITIVE_FACILITIES = POTENTIAL_LOCATION // 3
        AVAILABLE_LOCATIONS = POTENTIAL_LOCATION - EXISTING_COMPETITIVE_FACILITIES

        BETA = beta  # the distance sensitivity parameter
        LAMBDA = lamda  # the elasticity parameter
        THETA = theta  # sensitivity parameter of the utility function

        N = [node for node in range(1, CUSTOMERS + 1)]  # index of customers
        P = random.sample(N, POTENTIAL_LOCATION)  # index of potential locations
        C = random.sample(P, EXISTING_COMPETITIVE_FACILITIES)  # index of competitive facility locations
        S = [facility for facility in P if facility not in C]  # index of controlled facilities

        locations = {}
        for i in N:
            x = random.randint(1, MAP_SIZE - 1)
            y = random.randint(1, MAP_SIZE - 1)

            while (x, y) in locations.values():
                x = random.randint(1, MAP_SIZE - 1)
                y = random.randint(1, MAP_SIZE - 1)
            locations.update({i: (x, y)})

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

        w = {}
        for i in N:
            w.update({i: random.randint(1, 5)})

        c_attractiveness = {}
        for c in C:
            c_attractiveness.update({c: random.randint(3, 5)})

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

        def get_u_c(customer):
            utility_sum = 0
            for c in C:
                utility_sum += c_attractiveness.get(c) * ((distances.get((customer, c)) + 1) ** (-BETA))
            return utility_sum

        def get_omega(utility, customer):
            return get_g(utility + get_u_c(customer)) * (1 - (get_u_c(customer) / (utility + get_u_c(customer))))

        x_appr = result_approximation[0]
        x_exact = result_exact[0]

        obj_appr = 0.0
        obj_exact = 0.0
        for index, row in x_appr.iterrows():
            for i in N:
                # print(get_omega(get_utility(i, row["j"], row["r"]), i))
                obj_appr += w[i] * get_omega(get_utility(i, row["j"], row["r"]), i)
        for index, row in x_exact.iterrows():
            for i in N:
                # print(get_omega(get_utility(i, row["j"], row["r"]), i))
                obj_exact += w[i] * get_omega(get_utility(i, row["j"], row["r"]), i)
        print(S)
        print(result_approximation)
        print(result_exact)
        print("Obj. Exact: " + str(obj_exact) + " -- " + "Obj. Appr: " + str(obj_appr))
        print("Relative Error: " + str((obj_exact - obj_appr) / obj_exact) + "  vs. Alpha: " + str(epsilon))
        return (obj_exact - obj_appr) / obj_exact


    result_df = pd.DataFrame(columns=["N","Seed","Lambda","Beta","Theta","Relative Error 5%","TLA.Time 5%","IP.Time 5%", "Total.TLA.Time 5%","Relative Error 1%","TLA.Time 1%","IP.Time 1%", "Total.TLA.Time 1%","Exact.Time","Num.Facilities"])
    # for i in [80,90,100,110,120,130,140,160,180,200,250,300,350,400,450,500,550,600]:
    for i in [80,90,100,110,120,130,140,160,180,200,250,300,350,400]:
        for s in range(100, 105):
            beta = 1
            lamda = 1
            theta = 1
            seed = s
            result_approx_5 = tla(i, 0.05, beta, lamda, theta, seed)
            result_approx_1 = tla(i, 0.01, beta, lamda, theta, seed)
            result_exact = exact(i, beta, lamda, theta, seed)
            rel_err_5 = result(i, 0.05, beta, lamda, theta, result_approx_5, result_exact, seed)
            rel_err_1 = result(i, 0.01, beta, lamda, theta, result_approx_1, result_exact, seed)
            total_time_TLA_5 = round(result_approx_5[1] + result_approx_5[2], 2)
            total_time_TLA_1 = round(result_approx_1[1] + result_approx_1[2], 2)    
            result_df.loc[len(result_df.index)] = [i,seed,lamda,beta,theta,rel_err_5,result_approx_5[1],result_approx_5[2],total_time_TLA_5,rel_err_1,result_approx_1[1],result_approx_1[2],total_time_TLA_1,result_exact[1],result_exact[2]]
        
    print(result_df)
    result_df.to_csv("result1.csv", encoding='utf-8')