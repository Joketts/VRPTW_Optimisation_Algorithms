import os
import matplotlib.pyplot as plt
from benchmark_loader import VRPBenchmarkLoader
from plot_route import plot_algo_route

def run_benchmarks(dataset_type="solomon", algorithm="sa", generations=500):
    # Define the base path for the given dataset type.
    base_path = "./datasets/solomon_100" if dataset_type.lower() == "solomon" else "./datasets/homberger_800_customer"

    # List all .txt files in the dataset folder.
    files = sorted(set(f.lower() for f in os.listdir(base_path) if f.endswith(".txt")))
    print("Files found:", files)

    # Dictionary to store the performance metric (e.g., total route distance) per benchmark file.
    results = {}

    for file in files:
        # Remove the file extension to get the dataset name.
        dataset_name = file[:-4]
        print(f"\nRunning benchmark for dataset {dataset_name} using {algorithm.upper()} algorithm...")

        # Load the dataset.
        loader = VRPBenchmarkLoader(dataset_type=dataset_type, dataset_name=dataset_name)
        data = loader.load_data()

        # Run the selected algorithm.
        if algorithm.lower() == "ga":
            from GA import VRPGeneticAlgorithm
            algo = VRPGeneticAlgorithm(
                customers=data["customers"],
                vehicle_info=data["vehicle_info"],
                generations=generations
            )
            best_solution = algo.run_evolution()
            # Calculate the total route distance using the GA’s method.
            total_distance = algo.calculate_route_distance(best_solution)
        elif algorithm.lower() == "sa":
            from SA import VRPSimulatedAnnealing
            algo = VRPSimulatedAnnealing(
                customers=data["customers"],
                vehicle_info=data["vehicle_info"],
                initial_temp=1000,
                final_temp=1,
                alpha=0.998,
                iterations_per_temp=150
            )
            best_solution = algo.run_simulated_annealing()
            # For SA, sum distances for each route.
            total_distance = sum(algo.calculate_route_distance(route) for route in best_solution)
        else:
            raise ValueError("Unsupported algorithm. Choose 'ga' or 'sa'.")

        # Store the performance metric.
        results[dataset_name] = total_distance

        # Optionally, visualize the route for each dataset.
        #plot_algo_route(data["customers"], best_solution)

    # Plot aggregated benchmark results.
    dataset_names = list(results.keys())
    distances = list(results.values())

    plt.figure(figsize=(10, 6))
    plt.bar(dataset_names, distances)
    plt.xlabel("Benchmark Dataset")
    plt.ylabel("Total Route Distance")
    plt.title(f"Benchmark Results for {dataset_type.capitalize()} using {algorithm.upper()}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Example usage: set algorithm="sa" to run SA for all datasets.
run_benchmarks(dataset_type="solomon", algorithm="sa", generations=500)

