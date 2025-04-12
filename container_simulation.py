import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt

# Container multipliers
containers = {
    "Container 1": 10,
    "Container 2": 80,
    "Container 3": 31,
    "Container 4": 37,
    "Container 5": 17,
    "Container 6": 90,
    "Container 7": 50,
    "Container 8": 20,
    "Container 9": 73,
    "Container 10": 89
}

# Base treasure
BASE_TREASURE = 10000

# Cost of opening second container
SECOND_CONTAINER_COST = 50000

# Maximum number of inhabitants per container
MAX_INHABITANTS = 10

def calculate_profit(container_choices, all_player_choices, container_inhabitants):
    """
    Calculate profit based on container choices.
    
    Args:
        container_choices: List of containers chosen by player
        all_player_choices: List of all container choices made by all players
        container_inhabitants: Dict of inhabitants per container
    
    Returns:
        Total profit
    """
    total_profit = 0
    
    for i, container in enumerate(container_choices):
        # Calculate number of times this container was chosen
        container_openings = all_player_choices.count(container)
        
        # Calculate percentage of total openings
        total_openings = len(all_player_choices)
        opening_percentage = container_openings / total_openings
        
        # Calculate profit
        treasure = BASE_TREASURE * containers[container]
        divisor = container_inhabitants[container] + opening_percentage * 100  # * 100 to convert to actual percentage
        container_profit = treasure / divisor
        
        # Apply cost for second container
        if i == 1:  # Second container
            container_profit -= SECOND_CONTAINER_COST
            
        total_profit += container_profit
    
    return total_profit

def run_simulation(num_simulations=10000, num_players=100):
    """
    Run Monte Carlo simulation for container selection.
    
    Args:
        num_simulations: Number of simulations to run
        num_players: Number of players in each simulation
    
    Returns:
        DataFrame with average profits for each container combination
    """
    container_names = list(containers.keys())
    combinations_list = list(combinations(container_names, 2))
    
    # Initialize results dictionary
    results = {f"{c1},{c2}": [] for c1, c2 in combinations_list}
    
    for _ in range(num_simulations):
        # Randomly assign inhabitants to containers (1-10 per container)
        container_inhabitants = {container: np.random.randint(1, MAX_INHABITANTS + 1) 
                                for container in container_names}
        
        # Simulate other players choosing containers
        # Each player chooses 1-2 containers
        all_player_choices = []
        for _ in range(num_players):
            num_containers_to_choose = np.random.randint(1, 3)  # 1 or 2
            player_choices = np.random.choice(container_names, num_containers_to_choose, replace=False)
            all_player_choices.extend(player_choices)
        
        # Calculate profit for each possible container combination
        for c1, c2 in combinations_list:
            # Calculate profit for choosing these containers
            profit = calculate_profit([c1, c2], all_player_choices, container_inhabitants)
            results[f"{c1},{c2}"].append(profit)
    
    # Calculate average profit for each combination
    avg_results = {combo: np.mean(profits) for combo, profits in results.items()}
    
    # Convert to DataFrame for better visualization
    result_df = pd.DataFrame([
        {
            'First Container': combo.split(',')[0],
            'Second Container': combo.split(',')[1],
            'Average Profit': avg_results[combo],
            'Standard Deviation': np.std(results[combo]),
            'Probability of Profit > 0': np.mean([p > 0 for p in results[combo]])
        }
        for combo in results.keys()
    ])
    
    # Sort by average profit
    result_df = result_df.sort_values('Average Profit', ascending=False)
    
    return result_df, results

def analyze_single_containers(num_simulations=10000, num_players=100):
    """
    Analyze the expected profit from choosing just one container.
    """
    container_names = list(containers.keys())
    results = {container: [] for container in container_names}
    
    for _ in range(num_simulations):
        # Randomly assign inhabitants to containers (1-10 per container)
        container_inhabitants = {container: np.random.randint(1, MAX_INHABITANTS + 1) 
                                for container in container_names}
        
        # Simulate other players choosing containers
        all_player_choices = []
        for _ in range(num_players):
            num_containers_to_choose = np.random.randint(1, 3)  # 1 or 2
            player_choices = np.random.choice(container_names, num_containers_to_choose, replace=False)
            all_player_choices.extend(player_choices)
        
        # Calculate profit for each container
        for container in container_names:
            profit = calculate_profit([container], all_player_choices, container_inhabitants)
            results[container].append(profit)
    
    # Calculate average profit for each container
    single_container_df = pd.DataFrame([
        {
            'Container': container,
            'Multiplier': containers[container],
            'Average Profit': np.mean(profits),
            'Standard Deviation': np.std(profits)
        }
        for container, profits in results.items()
    ])
    
    # Sort by average profit
    single_container_df = single_container_df.sort_values('Average Profit', ascending=False)
    
    return single_container_df

def plot_results(results_dict):
    """Plot distribution of profits for top combinations"""
    plt.figure(figsize=(15, 10))
    
    # Get top 5 combinations by average profit
    avg_profits = {combo: np.mean(profits) for combo, profits in results_dict.items()}
    top_combos = sorted(avg_profits.keys(), key=lambda x: avg_profits[x], reverse=True)[:5]
    
    for combo in top_combos:
        plt.hist(results_dict[combo], alpha=0.5, bins=50, label=combo)
    
    plt.xlabel('Profit')
    plt.ylabel('Frequency')
    plt.title('Distribution of Profits for Top 5 Container Combinations')
    plt.legend()
    plt.grid(True)
    plt.savefig('profit_distribution.png')
    
    return 'profit_distribution.png'

def main():
    print("Running container selection simulation...")
    
    # Analyze single container profits
    print("\nAnalyzing single container profits:")
    single_results = analyze_single_containers()
    print(single_results)
    
    # Run simulation for combinations
    results_df, results_dict = run_simulation()
    
    print("\nTop 10 container combinations by average profit:")
    print(results_df.head(10))
    
    print("\nBottom 5 container combinations by average profit:")
    print(results_df.tail(5))
    
    # Identify best strategy
    best_combo = results_df.iloc[0]
    print(f"\nBest strategy: Choose {best_combo['First Container']} first (free), then {best_combo['Second Container']} (costs {SECOND_CONTAINER_COST})")
    print(f"Expected profit: {best_combo['Average Profit']:.2f}")
    print(f"Probability of positive profit: {best_combo['Probability of Profit > 0']:.2%}")
    
    # Check if it's better to choose just one container
    best_single = single_results.iloc[0]
    print(f"\nBest single container: {best_single['Container']}")
    print(f"Expected profit from choosing only one container: {best_single['Average Profit']:.2f}")
    
    if best_single['Average Profit'] > best_combo['Average Profit']:
        print("\nRECOMMENDATION: Choose only one container.")
    else:
        print("\nRECOMMENDATION: Choose two containers as per the best strategy.")
    
    # Plot results
    plot_path = plot_results(results_dict)
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    main() 