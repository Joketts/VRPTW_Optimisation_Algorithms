import random
import math
import numpy as np
import pandas as pd
import copy
from benchmark_loader import VRPBenchmarkLoader
from plot_route import plot_algo_route


# Load the Solomon dataset
solomon_loader = VRPBenchmarkLoader(dataset_type="solomon", dataset_name="c102")
solomon_data = solomon_loader.load_data()

class VRPSimulatedAnnealing:
    def __init__(self, customers, vehicle_info, initial_temp=1000, final_temp=1, alpha=0.998, iterations_per_temp=100):
        """
        Initializes the SA algorithm for a multi-vehicle VRP.
        :param customers: DataFrame with customer info.
        :param vehicle_info: Dict with vehicle info (e.g., {'num_vehicles': 10, 'capacity': 200}).
        :param initial_temp: Starting temperature.
        :param final_temp: Stopping temperature.
        :param alpha: Cooling rate.
        :param iterations_per_temp: Iterations per temperature.
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
        num_customers = len(self.customers)
        distance_matrix = np.zeros((num_customers, num_customers))
        for i in range(num_customers):
            for j in range(num_customers):
                if i != j:
                    dx = self.customers.loc[i, "x"] - self.customers.loc[j, "x"]
                    dy = self.customers.loc[i, "y"] - self.customers.loc[j, "y"]
                    distance_matrix[i][j] = math.sqrt(dx * dx + dy * dy)
        return distance_matrix

    def create_random_solution(self):
        """
        Generates an initial solution as a list of routes.
        Each route is a list starting and ending with depot (assumed id=0).
        Customers are randomly partitioned among vehicles.
        """
        customer_ids = list(self.customers["id"].values)[1:]  # Exclude depot
        random.shuffle(customer_ids)
        solution = []
        k = self.num_vehicles
        segment_length = math.ceil(len(customer_ids) / k)
        for i in range(0, len(customer_ids), segment_length):
            segment = customer_ids[i:i+segment_length]
            route = [0] + segment + [0]
            solution.append(route)
        # If there are fewer routes than vehicles, add empty routes [0,0]
        while len(solution) < k:
            solution.append([0, 0])
        return solution

    def calculate_route_distance(self, route):
        """Calculates the total distance of a single route."""
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
            for customer in route[1:]:
                if customer == 0:
                    current_time = 0
                    continue
                ready, due, service = self.time_windows[customer]
                if current_time < ready:
                    current_time = ready
                lateness = max(0, current_time - due)
                route_lateness += lateness**2  # or lateness**1.5, etc.
                current_time += service

            # Add a moderate route penalty if route_lateness > 0
            # Instead of 1e9, do e.g. route_lateness * 1000
            route_penalty = 1000 * route_lateness
            total_cost += route_distance + route_penalty
        return total_cost

    def generate_neighbor_solution(self, solution):
        """
        Generates a neighbor solution by either performing an intra-route move (within one route)
        or an inter-route move (moving a customer between routes).
        """
        new_solution = copy.deepcopy(solution)
        move_type = random.choice(["intra", "inter"])

        if move_type == "intra":
            # Choose a random route with at least one customer (excluding depot)
            route_idx = random.choice([i for i, route in enumerate(new_solution) if len(route) > 2])
            route = new_solution[route_idx]
            # Choose a random intra-route operator: swap, reinsert, or reverse.
            operator = random.choice(["swap", "reinsert", "reverse"])
            if operator == "swap" and len(route) > 3:
                i, j = random.sample(range(1, len(route) - 1), 2)
                route[i], route[j] = route[j], route[i]
            elif operator == "reinsert" and len(route) > 2:
                i = random.randint(1, len(route) - 2)
                customer = route.pop(i)
                j = random.randint(1, len(route) - 1)
                route.insert(j, customer)
            elif operator == "reverse" and len(route) > 3:
                i, j = sorted(random.sample(range(1, len(route) - 1), 2))
                route[i:j] = route[i:j][::-1]
            new_solution[route_idx] = route
        else:  # inter-route move
            # Choose two distinct routes
            valid_routes = [i for i, route in enumerate(new_solution) if len(route) > 2]
            if len(valid_routes) >= 2:
                r1, r2 = random.sample(valid_routes, 2)
                route1, route2 = new_solution[r1], new_solution[r2]
                # Move a random customer from route1 to a random position in route2
                idx = random.randint(1, len(route1) - 2)
                customer = route1.pop(idx)
                insert_pos = random.randint(1, len(route2) - 1)
                route2.insert(insert_pos, customer)
                new_solution[r1] = route1
                new_solution[r2] = route2
            # Otherwise, fall back to an intra-route move
        return new_solution

    def has_violation(self, route):
        """Checks if a single route has any time window violations."""
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
        Repairs a single route by moving the customer with the worst violation
        to an earlier position.
        """
        current_time = 0
        worst_violation = 0
        worst_index = None
        for i in range(1, len(route) - 1):
            customer = route[i]
            ready, due, service = self.time_windows[customer]
            if current_time < ready:
                current_time = ready
            lateness = max(0, current_time - due)
            if lateness > worst_violation:
                worst_violation = lateness
                worst_index = i
            current_time += service

        if worst_index is not None and worst_index > 1:
            new_route = route[:]
            customer = new_route.pop(worst_index)
            new_pos = random.randint(1, worst_index - 1)
            new_route.insert(new_pos, customer)
            return new_route
        return route

    def repair_solution(self, solution):
        """
        Repairs each route in the solution if it has time window violations.
        """
        new_solution = []
        for route in solution:
            if self.has_violation(route):
                new_solution.append(self.repair_route(route))
            else:
                new_solution.append(route)
        return new_solution

    def run_simulated_annealing(self):
        """Runs the SA algorithm and returns the best multi-route solution found."""
        current_solution = self.create_random_solution()
        current_cost = self.cost_function(current_solution)
        best_solution = copy.deepcopy(current_solution)
        best_cost = current_cost

        T = self.initial_temp
        iteration = 0

        while T > self.final_temp:
            for _ in range(self.iterations_per_temp):
                neighbor = self.generate_neighbor_solution(current_solution)
                # Optionally repair the neighbor solution if it has violations.
                neighbor = self.repair_solution(neighbor)
                neighbor_cost = self.cost_function(neighbor)
                delta = neighbor_cost - current_cost

                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_solution = neighbor
                    current_cost = neighbor_cost
                    if current_cost < best_cost:
                        best_solution = copy.deepcopy(current_solution)
                        best_cost = current_cost

            if iteration % 100 == 0:
                print(f"Iteration {iteration}, T = {T:.2f}, Best Cost = {best_cost:.2f}")
            T *= self.alpha
            iteration += 1

        print("\nFinal Best Cost:", best_cost)
        print("Final Best Solution (routes):")
        for r in best_solution:
            print(r)
        self.check_solution_violations(best_solution)

        # Calculate and print the total distance of the solution
        total_distance = sum(self.calculate_route_distance(route) for route in best_solution)
        print("Total Distance:", total_distance)

        return best_solution

    def check_solution_violations(self, solution):
        """Prints time window violations for each route in the solution."""
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
            if violations:
                print("Route", route, "has violations:")
                for v in sorted(violations, key=lambda x: x[3], reverse=True)[:5]:
                    print(f" Customer {v[0]}: Arrived {v[1]}, due {v[2]}, late by {v[3]}")
            else:
                print("Route", route, "satisfies all time window constraints.")


# Run the multi-vehicle SA algorithm
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
plot_algo_route(solomon_data["customers"], best_solution_sa)
