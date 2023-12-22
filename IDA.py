import random
import timeit

class State:
    def __init__(self, num_cities):
        self.visited = [False] * num_cities
        self.num_visited = 0
        self.current_id = 0

def generate_random_tsp_instance(num_cities, seed):
    random.seed(seed)
    costs = [[0 if i == j else random.randint(1, 100) for j in range(num_cities)] for i in range(num_cities)]
    return costs

def min_out_heuristic(state, costs):

        return 0


def ida_star(state, heuristic_func, costs, time_limit):
    start_time = timeit.default_timer()
    path = [0]  # Start from city 0
    bound = heuristic_func(state, costs)
    generated_nodes = 0
    expanded_nodes = 0

    def search(state, g, bound, path, heuristic_func, costs, time_limit):
        nonlocal generated_nodes, expanded_nodes
        h = heuristic_func(state, costs)
        f = g + h
        if f > bound:
            return f, False, path
        if state.num_visited == len(costs) and state.current_id == 0 and state.visited[0]:
            return f, True, path
        min_cost = float('inf')
        expanded = False  
        for next_city in range(len(costs)):
            if not state.visited[next_city]:
                new_state = State(len(costs))
                new_state.visited = state.visited.copy()
                new_state.visited[next_city] = True
                new_state.num_visited = state.num_visited + 1
                new_state.current_id = next_city

                new_path = path + [next_city]
                generated_nodes += 1
                t, found, updated_path = search(new_state, g + costs[state.current_id][next_city], bound, new_path,
                                               heuristic_func, costs, time_limit)
                if found:
                    return t, True, updated_path
                if t < min_cost:
                    min_cost = t
                expanded = True
        if expanded == True:
            expanded_nodes += 1
        return min_cost, False, path

    while True:
        t, found, new_path = search(state, 0, bound, path, heuristic_func, costs, time_limit)
        if found:
            end_time = timeit.default_timer()
            run_time = end_time - start_time
            return run_time, t, new_path, expanded_nodes, generated_nodes
        if t == float('inf'):
            print("No solution found within the time limit.")
            return None, None, None, None, None
        bound = t

# Run experiments for multiple instances with the same seed for each N
num_cities_list = [5, 10, 11, 12]
num_instances = 5
time_limit = 20 * 60  # 20 minutes in seconds

for num_cities in num_cities_list:
    for instance in range(1, num_instances + 1):
        seed = instance
        costs = generate_random_tsp_instance(num_cities, seed)
        initial_state = State(num_cities)

        run_time, optimal_cost, optimal_path, expanded_nodes, generated_nodes = ida_star(
            initial_state, min_out_heuristic, costs, time_limit
        )

        if run_time is not None and optimal_cost is not None:
            print(f"Results for {num_cities} cities (Seed {seed}):")
            print(f"Run Time: {run_time:.6f} seconds")
            print(f"Optimal Path Cost: {optimal_cost:.2f}")
            print(f"Optimal Path: {optimal_path}")
            print(f"Number of Expanded Nodes: {expanded_nodes}")
            print(f"Number of Generated Nodes: {generated_nodes}")
            print()


