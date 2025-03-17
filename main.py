from src.ga_solver import run_ga
from src.sa_solver import run_sa

def main():
    print("Running Genetic Algorithm...")
    ga_result = run_ga()

    print("\nRunning Simulated Annealing...")
    sa_result = run_sa()

    print("\nComparison Complete!")

if __name__ == "__main__":
    main()
