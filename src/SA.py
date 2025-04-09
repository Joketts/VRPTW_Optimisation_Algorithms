import random
import math
import numpy as np
import copy
from benchmark_loader import VRPBenchmarkLoader
from plot_route import plot_algo_route


# load the dataset
solomon_loader = VRPBenchmarkLoader(dataset_type="solomon", dataset_name="r106")
solomon_data = solomon_loader.load_data()

class VRPSimulatedAnnealing:
    def __init__(self, customers, vehicle_info, initial_temp=1000, final_temp=1, alpha=0.998, iterations_per_temp=100):
        """
        Initializes the SA
        """
        self.customers = customers
        self.vehicle_info = vehicle_info
        self.num_vehicles = vehicle_info.get('num_vehicles', 1)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.iterations_per_temp = iterations_per_temp

        self.distance_matrix = self.precompute_distances()
        self.time_windows = {
            row["id"]: (row["ready_time"], row["due_time"], row["service_time"])
            for _, row in self.customers.iterrows()
        }

    def precompute_distances(self):
        # gets numb of customers
        num_customers = len(self.customers)

        # init 2d Numpy array fill w 0's
        # stores computed distances between pairs
        distance_matrix = np.zeros((num_customers, num_customers))

        # loop over each customer index to compute pairwise distances
        for i in range(num_customers):
            for j in range(num_customers):
                # skip computation if comparing same customer
                if i != j:
                    # get x, y coords
                    dx = self.customers.loc[i, "x"] - self.customers.loc[j, "x"]
                    dy = self.customers.loc[i, "y"] - self.customers.loc[j, "y"]
                    # get euclidean distance
                    distance_matrix[i][j] = math.sqrt(dx * dx + dy * dy)
        return distance_matrix

    def create_random_solution(self):
        """
        initial solution, list of routes
        starts ends w depo
        customers randomly put into list
        """
        # get customer id from dataframe, ignore depo
        customer_ids = list(self.customers["id"].values)[1:]
        # shuffle customer id to get random order
        random.shuffle(customer_ids)
        solution = []
        # get numb of vehicles
        k = self.num_vehicles
        # get length of each route
        segment_length = math.ceil(len(customer_ids) / k)

        # run through customer ids in chunks of size segment length
        for i in range(0, len(customer_ids), segment_length):
            segment = customer_ids[i:i+segment_length]
            route = [0] + segment + [0]
            # adds route to list
            solution.append(route)
        # if we didn't use every vehicle set empty to 0,0
        while len(solution) < k:
            solution.append([0, 0])
        return solution

    def calculate_route_distance(self, route):
        """ calculates length of route"""
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += self.distance_matrix[route[i], route[i + 1]]
        return total_distance

    def cost_function(self, solution):
        total_cost = 0
        for route in solution:
            route_distance = self.calculate_route_distance(route)
            route_lateness = 0
            current_time = 0
            # interates over customer start at second point ( miss depo)
            for customer in route[1:]:
                if customer == 0:
                    current_time = 0
                    continue
                    # get time window contraints for customer
                ready, due, service = self.time_windows[customer]
                # waits for ready time
                if current_time < ready:
                    current_time = ready
                lateness = max(0, current_time - due)
                # lateness penalized
                route_lateness += lateness**2
                current_time += service

            # calculate penalty for lateness
            route_penalty = 1000 * route_lateness
            total_cost += route_distance + route_penalty
        return total_cost

    def generate_neighbor_solution(self, solution):
        """
        generates neighbor solution by moving within the route or moving custer between routes
        """
        new_solution = copy.deepcopy(solution)
        move_type = random.choice(["intra", "inter"])

        if move_type == "intra":
            # get route with more than just depo in it
            route_idx = random.choice([i for i, route in enumerate(new_solution) if len(route) > 2])
            route = new_solution[route_idx]
            # chose operator at random
            operator = random.choice(["swap", "reinsert", "reverse"])

            if operator == "swap" and len(route) > 3:
                # selects two pos (not depo)
                # swaps customer
                i, j = random.sample(range(1, len(route) - 1), 2)
                route[i], route[j] = route[j], route[i]
            elif operator == "reinsert" and len(route) > 2:
                # remove customer rnd pos
                # insert rnd pos
                i = random.randint(1, len(route) - 2)
                customer = route.pop(i)
                j = random.randint(1, len(route) - 1)
                route.insert(j, customer)
            elif operator == "reverse" and len(route) > 3:
                # select apart of route reverse order
                i, j = sorted(random.sample(range(1, len(route) - 1), 2))
                route[i:j] = route[i:j][::-1]
                # updates changed route back into solution
            new_solution[route_idx] = route
        else:  # inter-route move
            # Choose two distinct routes
            valid_routes = [i for i, route in enumerate(new_solution) if len(route) > 2]
            if len(valid_routes) >= 2:
                # rnd picks two routes
                r1, r2 = random.sample(valid_routes, 2)
                route1, route2 = new_solution[r1], new_solution[r2]

                # selects rnd customer from first route
                idx = random.randint(1, len(route1) - 2)
                customer = route1.pop(idx)
                # inserts customer into rnd pos in second route
                insert_pos = random.randint(1, len(route2) - 1)
                route2.insert(insert_pos, customer)
                # updates routes in solution
                new_solution[r1] = route1
                new_solution[r2] = route2
            # isn't enough valid routes, fall back
        return new_solution

    def has_violation(self, route):
        """checks single route for violations"""
        current_time = 0
        for customer in route[1:]:
            if customer == 0:
                current_time = 0
                continue
            ready, due, service = self.time_windows[customer]
            if current_time < ready:
                current_time = ready
            if current_time > due:
                return True
            current_time += service
        return False

    def repair_route(self, route):
        """
        repairs single route by moving custer with worse violation to earlier pos
        """
        current_time = 0
        worst_violation = 0
        worst_index = None
        # loop through route customers
        for i in range(1, len(route) - 1):
            customer = route[i]
            ready, due, service = self.time_windows[customer]
            if current_time < ready:
                current_time = ready
            lateness = max(0, current_time - due)
            # if lateness is greater than worst found
            # record customer id
            if lateness > worst_violation:
                worst_violation = lateness
                worst_index = i
            current_time += service

        # customer not in pos 1 n violation
        if worst_index is not None and worst_index > 1:
            # create copy
            new_route = route[:]
            # remove customer
            customer = new_route.pop(worst_index)
            # randomly reinsert customer
            new_pos = random.randint(1, worst_index - 1)
            new_route.insert(new_pos, customer)
            return new_route
        return route

    def repair_solution(self, solution):
        """
        repairs each route in the solution if it has time window violations
        """
        new_solution = []
        for route in solution:
            if self.has_violation(route):
                new_solution.append(self.repair_route(route))
            else:
                new_solution.append(route)
        return new_solution

    def run_simulated_annealing(self):

        # generates initial solution rnd
        current_solution = self.create_random_solution()
        # evaulate cost of solution
        current_cost = self.cost_function(current_solution)
        # initalize best_solution w init solution
        best_solution = copy.deepcopy(current_solution)
        best_cost = current_cost

        # set starting temp
        T = self.initial_temp
        iteration = 0

        # run until temp = final temp
        while T > self.final_temp:
            # for each temp level run set amount of iterations
            for _ in range(self.iterations_per_temp):
                # generate neighbor from current solution
                neighbor = self.generate_neighbor_solution(current_solution)
                # repair the neighbor to fix any violations
                neighbor = self.repair_solution(neighbor)
                # calculate cost of new neighbor
                neighbor_cost = self.cost_function(neighbor)
                # compute difference in cost between the two solutions
                delta = neighbor_cost - current_cost

                # accept new neighbor if it has lower cost or if its accepted probabilistically
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_solution = neighbor
                    current_cost = neighbor_cost
                    if current_cost < best_cost:
                        best_solution = copy.deepcopy(current_solution)
                        best_cost = current_cost

            if iteration % 100 == 0:
                print(f"Iteration {iteration}, T = {T:.2f}, Best Cost = {best_cost:.2f}")

            # cool temp by multiplying by alpha
            T *= self.alpha
            iteration += 1

        #print("\nFinal Best Cost:", best_cost)
        #print("Final Best Solution (routes):")
        # print best solution
        for r in best_solution:
            print(r)

        # check solution violates contraints
        self.check_solution_violations(best_solution)

        # calculate distance total distance
        total_distance = sum(self.calculate_route_distance(route) for route in best_solution)
        print("Total Distance:", total_distance)

        return best_solution

    def check_solution_violations(self, solution):
        """prints time window violations"""
        for route in solution:
            current_time = 0
            violations = []
            for customer in route[1:]:
                if customer == 0:
                    current_time = 0
                    continue
                ready, due, service = self.time_windows[customer]
                if current_time < ready:
                    current_time = ready
                if current_time > due:
                    violations.append((customer, current_time, due, current_time - due))
                current_time += service
            # if violations:
            #     print("Route", route, "has violations:")
            #     for v in sorted(violations, key=lambda x: x[3], reverse=True)[:5]:
            #         print(f" Customer {v[0]}: Arrived {v[1]}, due {v[2]}, late by {v[3]}")
            # else:
            #     print("Route", route, "satisfies all time window constraints.")


# runs algorithms
vrp_sa_multi = VRPSimulatedAnnealing(
    customers=solomon_data["customers"],
    vehicle_info=solomon_data["vehicle_info"],
    initial_temp=1000,
    final_temp=1,
    alpha=0.998,
    iterations_per_temp=150
)

best_solution_sa = vrp_sa_multi.run_simulated_annealing()

print("\nBest Route Found by SA:", best_solution_sa)
#plots route
plot_algo_route(solomon_data["customers"], best_solution_sa)
