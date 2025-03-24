import random
import pandas as pd
import numpy as np
from benchmark_loader import VRPBenchmarkLoader
from plot_route import plot_algo_route

solomon_loader = VRPBenchmarkLoader(dataset_type="solomon", dataset_name="c102")
solomon_data = solomon_loader.load_data()

class VRPGeneticAlgorithm:
    def __init__(self, customers, vehicle_info, population_size=500, generations=100, mutation_rate=0.4):
        self.customers = customers.reset_index(drop=True)  # Reset index for safe matrix lookups
        self.vehicle_info = vehicle_info
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.num_vehicles = vehicle_info["num_vehicles"]

        # Proper ID → row index mapping
        self.id_to_index = dict(zip(self.customers["id"], self.customers.index))

        # Precompute distance matrix
        self.distance_matrix = self.precompute_distances()

        # Precompute time window lookups
        self.time_windows = {
            row["id"]: (row["ready_time"], row["due_time"], row["service_time"])
            for _, row in self.customers.iterrows()
        }

        self.population = self.initialize_population()

    def precompute_distances(self):
        num_customers = len(self.customers)
        dist = np.zeros((num_customers, num_customers))
        for i in range(num_customers):
            for j in range(num_customers):
                if i != j:
                    dist[i][j] = np.hypot(
                        self.customers.loc[i, "x"] - self.customers.loc[j, "x"],
                        self.customers.loc[i, "y"] - self.customers.loc[j, "y"]
                    )
        return dist

    def initialize_population(self):
        customers = list(self.customers["id"].values[1:])  # exclude depot
        population = []
        for _ in range(self.population_size):
            random.shuffle(customers)
            routes = [[] for _ in range(self.num_vehicles)]
            for idx, cust in enumerate(customers):
                routes[idx % self.num_vehicles].append(cust)
            individual = [[0] + r + [0] for r in routes]
            population.append(individual)

        # print("Sample individual:", population[0])
        return population

    def calculate_route_distance(self, individual):
        total_distance = 0
        for route in individual:
            for i in range(len(route) - 1):
                from_id = int(route[i])
                to_id = int(route[i + 1])
                if from_id not in self.id_to_index or to_id not in self.id_to_index:
                    print(f"❌ ID mismatch: {from_id}, {to_id}")
                    continue
                from_idx = self.id_to_index[from_id]
                to_idx = self.id_to_index[to_id]
                total_distance += self.distance_matrix[from_idx][to_idx]
        return total_distance

    def fitness_function(self, individual):
        total_distance = self.calculate_route_distance(individual)
        penalty = 0
        is_feasible = True

        for route in individual:
            current_time = 0
            prev_customer = 0  # start from depot (customer 0)

            for cust in route[1:]:  # skip depot at start
                if cust == 0:
                    current_time = 0
                    prev_customer = 0
                    continue

                # Add travel time between previous and current customer
                travel_time = self.distance_matrix[prev_customer][cust]
                current_time += travel_time

                # Get time window for current customer
                ready, due, service = self.time_windows[int(cust)]

                # Wait if early
                if current_time < ready:
                    current_time = ready

                # Penalize if late
                if current_time > due:
                    lateness = current_time - due
                    penalty += (lateness ** 1.2) * 0.05  # scale down penalty
                    is_feasible = False

                # Add service time
                current_time += service
                prev_customer = cust

        # Bonus: Strong incentive for feasible routes
        bonus = 1e6 if is_feasible else 0

        # Fitness is inverse of total cost
        fitness = 1 / (total_distance + penalty + 1e-6)
        if is_feasible:
            fitness += 10  # or some strong positive boost

        #print(f"Distance: {total_distance:.2f} | Penalty: {penalty:.2f} | {'✅' if is_feasible else '❌'}")
        return fitness

    def tournament_selection(self, fitness_cache):
        candidates = random.sample(self.population, 5)
        feasibles = [c for c in candidates if fitness_cache[tuple(map(tuple, c))] > 1e-4]
        if feasibles:
            return min(feasibles, key=lambda c: fitness_cache[tuple(map(tuple, c))])
        return min(candidates, key=lambda c: fitness_cache[tuple(map(tuple, c))])

    def crossover(self, parent1, parent2):
        """Creates two valid children using ordered crossover and redistributes them across multiple vehicles."""
        parent1_flat = [c for route in parent1 for c in route if c != 0]
        parent2_flat = [c for route in parent2 for c in route if c != 0]
        size = len(parent1_flat)

        if size < 2:
            print("⚠️ Not enough customers to crossover.")
            return parent1, parent2

        start, end = sorted(random.sample(range(size), 2))

        # Child 1
        mid1 = parent1_flat[start:end]
        remaining1 = [c for c in parent2_flat if c not in mid1]
        child1_flat = remaining1[:start] + mid1 + remaining1[start:]

        # Child 2
        mid2 = parent2_flat[start:end]
        remaining2 = [c for c in parent1_flat if c not in mid2]
        child2_flat = remaining2[:start] + mid2 + remaining2[start:]

        # Redistribute to multi-vehicle routes
        def redistribute(customers_flat):
            total = len(customers_flat)
            base_size = total // self.num_vehicles
            extra = total % self.num_vehicles

            routes = []
            idx = 0
            for v in range(self.num_vehicles):
                count = base_size + (1 if v < extra else 0)
                segment = customers_flat[idx:idx+count]
                routes.append([0] + segment + [0] if segment else [0, 0])
                idx += count
            return routes

        child1 = redistribute(child1_flat)
        child2 = redistribute(child2_flat)

        # # Validation (optional debug)
        # flat1 = [c for r in child1 for c in r if c != 0]
        # flat2 = [c for r in child2 for c in r if c != 0]
        # all_customers = set(self.customers["id"].values[1:])
        #
        # if set(flat1) != all_customers:
        #     print("❌ Child 1 is missing customers or has duplicates!")
        # if set(flat2) != all_customers:
        #     print("❌ Child 2 is missing customers or has duplicates!")
        #
        # print(f"\n🧪 DEBUG CHILDREN (Gen {self.current_gen})")
        # print("Child 1:", child1)
        # print("Child 2:", child2)

        return child1, child2

    def mutate(self, individual):
        """Swaps two customers (between or within routes) while preserving all customers."""
        if random.random() > self.mutation_rate:
            return individual  # No mutation

        new = [route[:] for route in individual]  # Deep copy
        # Flatten customer positions (ignore depots)
        customer_positions = [(i, j)
                              for i, route in enumerate(new)
                              for j in range(1, len(route) - 1)]  # skip depot at 0 and -1

        if len(customer_positions) < 2:
            return individual  # Not enough to swap

        (r1, i1), (r2, i2) = random.sample(customer_positions, 2)
        new[r1][i1], new[r2][i2] = new[r2][i2], new[r1][i1]  # Swap

        return new

    def check_time_windows(self, individual):
        for route in individual:
            current_time = 0
            violations = []
            for cust in route[1:]:
                if cust == 0:
                    current_time = 0
                    continue
                ready, due, service = self.time_windows[cust]
                if current_time < ready:
                    current_time = ready
                if current_time > due:
                    violations.append((cust, current_time, due, current_time - due))
                current_time += service
            if violations:
                print(f"Route {route} has violations:")
                for v in violations:
                    print(f"  Customer {v[0]}: Arrived {v[1]}, due {v[2]}, late by {v[3]}")
            else:
                print(f"Route {route} satisfies all time window constraints.")

    def run_evolution(self):
        elite_size = int(self.population_size * 0.05)
        self.current_gen = 0

        for gen in range(self.generations):
            fitness_cache = {}
            for ind in self.population:
                key = tuple(map(tuple, ind))
                fit = self.fitness_function(ind)
                fitness_cache[key] = fit

            new_pop = sorted(self.population, key=lambda c: fitness_cache[tuple(map(tuple, c))])[:elite_size]

            while len(new_pop) < self.population_size:
                self.current_gen = gen
                p1 = self.tournament_selection(fitness_cache)
                p2 = self.tournament_selection(fitness_cache)

                c1, c2 = self.crossover(p1, p2)

                c1 = self.mutate(c1)
                c2 = self.mutate(c2)

                # Ensure they still contain all customers!
                flat1 = sorted([c for r in c1 for c in r if c != 0])
                flat2 = sorted([c for r in c2 for c in r if c != 0])
                expected = sorted(self.customers["id"].values[1:])

                if flat1 != list(expected):
                    print("❌ Child 1 is missing customers or has duplicates!")
                if flat2 != list(expected):
                    print("❌ Child 2 is missing customers or has duplicates!")

                new_pop.extend([c1, c2])

            self.population = new_pop

            log_interval = max(1, self.generations // 10)
            if gen % log_interval == 0:
                best = min(self.population, key=lambda c: fitness_cache.get(tuple(map(tuple, c)), self.fitness_function(c)))
                fitness = fitness_cache.get(tuple(map(tuple, best)), self.fitness_function(best))
                print(f"\nGeneration {gen}: Best Distance = {self.calculate_route_distance(best):.2f} | Fitness = {fitness:.6f}")

        best = min(self.population, key=lambda c: self.fitness_function(c))
        print(f"\nFinal Best Route Distance: {self.calculate_route_distance(best):.2f}")
        self.check_time_windows(best)
        return best


vrp_ga = VRPGeneticAlgorithm(
    customers=solomon_data["customers"],
    vehicle_info=solomon_data["vehicle_info"],
    generations=500
)

best_solution = vrp_ga.run_evolution()
print("\nBest Route Found:")
for route in best_solution:
    print(route)
plot_algo_route(solomon_data["customers"], best_solution)

