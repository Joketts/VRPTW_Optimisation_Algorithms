import random
import pandas as pd
import numpy as np
from benchmark_loader import VRPBenchmarkLoader
from plot_route import plot_algo_route

solomon_loader = VRPBenchmarkLoader(dataset_type="solomon", dataset_name="c102")
solomon_data = solomon_loader.load_data()

class VRPGeneticAlgorithm:
    def __init__(self, customers, vehicle_info, population_size=500, generations=100, mutation_rate=0.15):
        self.customers = customers.reset_index(drop=True)
        self.vehicle_info = vehicle_info
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.num_vehicles = vehicle_info["num_vehicles"]
        self.id_to_index = dict(zip(self.customers["id"], self.customers.index))
        self.distance_matrix = self.precompute_distances()
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
        customers = list(self.customers["id"].values[1:])
        customers.sort(key=lambda cid: self.time_windows[cid][0])
        population = []
        for _ in range(self.population_size):
            random.shuffle(customers)
            num_routes = random.randint(5, self.num_vehicles)
            routes = [[] for _ in range(num_routes)]
            for idx, cust in enumerate(customers):
                routes[idx % num_routes].append(cust)
            individual = [[0] + r + [0] for r in routes if r]
            population.append(individual)
        return population

    def edf_with_noise(self):
        tasks = [
            {
                'name': row['id'],
                'ready_time': row['ready_time'],
                'deadline': row['due_time'],
                'processing_time': row['service_time']
            }
            for _, row in self.customers.iterrows() if row['id'] != 0
        ]

        # Add noise to due times to diversify
        for task in tasks:
            task['deadline'] += random.randint(-30, 30)

        tasks_sorted = sorted(tasks, key=lambda t: t['deadline'])

        vehicles = [{'id': i, 'current_time': 0, 'route': [0]} for i in range(1, self.num_vehicles + 1)]

        for task in tasks_sorted:
            best_vehicle = min(vehicles, key=lambda v: max(v['current_time'], task['ready_time']) + task['processing_time'])
            best_vehicle['route'].append(task['name'])
            best_vehicle['current_time'] = max(best_vehicle['current_time'], task['ready_time']) + task['processing_time']

        for vehicle in vehicles:
            vehicle['route'].append(0)

        return [v['route'] for v in vehicles if len(v['route']) > 2]

    def calculate_route_distance(self, individual):
        total_distance = 0
        for route in individual:
            for i in range(len(route) - 1):
                from_id = int(route[i])
                to_id = int(route[i + 1])
                if from_id not in self.id_to_index or to_id not in self.id_to_index:
                    continue
                from_idx = self.id_to_index[from_id]
                to_idx = self.id_to_index[to_id]
                total_distance += self.distance_matrix[from_idx][to_idx]
        return total_distance

    def fitness_function(self, individual):
        total_distance = self.calculate_route_distance(individual)
        total_penalty = 0
        is_feasible = True

        for route in individual:
            current_time = 0
            prev_customer = 0
            for cust in route[1:]:
                if cust == 0:
                    current_time = 0
                    prev_customer = 0
                    continue
                travel_time = self.distance_matrix[self.id_to_index[prev_customer]][self.id_to_index[cust]]
                current_time += travel_time
                ready, due, service = self.time_windows[int(cust)]
                if current_time < ready:
                    current_time = ready
                if current_time > due:
                    lateness = current_time - due
                    total_penalty += lateness ** 2 * 1.5
                    is_feasible = False
                current_time += service
                prev_customer = cust

        num_routes = len(individual)
        vehicle_penalty = (num_routes - 10) ** 1.5 if num_routes > 10 else 0
        score = total_distance + total_penalty + vehicle_penalty + 1e-6
        fitness = 10000 / score
        if is_feasible:
            fitness *= 2
        return fitness

    def tournament_selection(self, fitnesses, k=3):
        selected_indices = random.sample(range(len(self.population)), k)
        candidates = []
        for i in selected_indices:
            ind = self.population[i]
            fitness = fitnesses.get(tuple(map(tuple, ind)), self.fitness_function(ind))
            feasible = fitness > 1.0
            candidates.append((i, fitness, feasible))

        feasible_candidates = [c for c in candidates if c[2]]
        if feasible_candidates:
            best = max(feasible_candidates, key=lambda x: x[1])
        else:
            best = max(candidates, key=lambda x: x[1])

        return self.population[best[0]]

    def crossover(self, parent1, parent2):
        parent1_flat = [c for route in parent1 for c in route if c != 0]
        parent2_flat = [c for route in parent2 for c in route if c != 0]
        size = len(parent1_flat)

        if size < 2:
            return parent1, parent2

        start, end = sorted(random.sample(range(size), 2))
        mid1 = parent1_flat[start:end]
        remaining1 = [c for c in parent2_flat if c not in mid1]
        child1_flat = remaining1[:start] + mid1 + remaining1[start:]

        mid2 = parent2_flat[start:end]
        remaining2 = [c for c in parent1_flat if c not in mid2]
        child2_flat = remaining2[:start] + mid2 + remaining2[start:]

        def redistribute(customers_flat):
            routes = [[] for _ in range(self.num_vehicles)]
            for i, cust in enumerate(customers_flat):
                routes[i % self.num_vehicles].append(cust)
            return [[0] + r + [0] for r in routes if r]

        return redistribute(child1_flat), redistribute(child2_flat)

    def mutate(self, individual):
        if random.random() > self.mutation_rate:
            return individual

        new = [route[:] for route in individual if len(route) > 2]
        if len(new) < 2:
            return individual

        original = [r[:] for r in new]

        if random.random() < 0.5:
            src, dst = random.sample(range(len(new)), 2)
            if len(new[src]) > 3:
                pos = random.randint(1, len(new[src]) - 2)
                cust = new[src].pop(pos)
                insert_pos = random.randint(1, len(new[dst]) - 1)
                new[dst].insert(insert_pos, cust)
        else:
            r1, r2 = random.sample(range(len(new)), 2)
            if len(new[r1]) > 2 and len(new[r2]) > 2:
                i = random.randint(1, len(new[r1]) - 2)
                j = random.randint(1, len(new[r2]) - 2)
                new[r1][i], new[r2][j] = new[r2][j], new[r1][i]

        for route in new:
            if random.random() < 0.05:
                middle = route[1:-1]
                random.shuffle(middle)
                route[1:-1] = middle

        if random.random() < 0.1 and len(new) > 1:
            new.sort(key=lambda r: self.calculate_route_distance([r]))
            merged = new[0][1:-1] + new[1][1:-1]
            new = new[2:] + [[0] + merged + [0]]

        return [r for r in new if len(r) > 2]

    def two_opt(self, route):
        best = route[:]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best) - 1):
                    if j - i == 1:
                        continue
                    new_route = best[:i] + best[i:j][::-1] + best[j:]
                    if self.calculate_route_distance([new_route]) < self.calculate_route_distance([best]):
                        best = new_route
                        improved = True
            if improved:
                break
        return best

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
        elite_size = int(self.population_size * 0.02)
        self.current_gen = 0
        best_fitness = -float('inf')
        no_improvement_counter = 0
        max_no_improve = 50

        for gen in range(self.generations):
            fitness_cache = {}
            for ind in self.population:
                key = tuple(map(tuple, ind))
                fit = self.fitness_function(ind)
                fitness_cache[key] = fit

            new_pop = sorted(self.population, key=lambda c: -fitness_cache[tuple(map(tuple, c))])[:elite_size]

            # Best tracking for stagnation
            current_best = max(self.population, key=lambda c: fitness_cache[tuple(map(tuple, c))])
            current_best_fit = fitness_cache[tuple(map(tuple, current_best))]

            if current_best_fit > best_fitness:
                best_fitness = current_best_fit
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= max_no_improve:
                print("\n⏹️ Stagnation detected — injecting new EDF-based individuals.")
                inject_count = int(0.1 * self.population_size)
                edf_individuals = [self.mutate(self.edf_with_noise()) for _ in range(inject_count)]
                new_pop.extend(edf_individuals)
                no_improvement_counter = 0  # reset counter after injection

            while len(new_pop) < self.population_size:
                self.current_gen = gen
                p1 = self.tournament_selection(fitness_cache)
                p2 = self.tournament_selection(fitness_cache)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                new_pop.extend([c1, c2])

            self.population = new_pop[:self.population_size]

            log_interval = max(1, self.generations // 10)
            if gen % log_interval == 0:
                best = max(self.population, key=lambda c: fitness_cache.get(tuple(map(tuple, c)), self.fitness_function(c)))
                fitness = fitness_cache.get(tuple(map(tuple, best)), self.fitness_function(best))
                print(f"\nGeneration {gen}: Best Distance = {self.calculate_route_distance(best):.2f} | Fitness = {fitness:.6f}")

        best = max(self.population, key=lambda c: self.fitness_function(c))
        print(f"\nFinal Best Route Distance: {self.calculate_route_distance(best):.2f}")
        self.check_time_windows(best)
        return best





vrp_ga = VRPGeneticAlgorithm(
    customers=solomon_data["customers"],
    vehicle_info=solomon_data["vehicle_info"],
    generations=350
)

best_solution = vrp_ga.run_evolution()
print("\nBest Route Found:")
for route in best_solution:
    print(route)
plot_algo_route(solomon_data["customers"], best_solution)
