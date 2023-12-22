import random
import timeit
import heapq

class State:
    def __init__(self, num_cities):
        # Initialize the state representation
        self.visited = [False] * num_cities
        self.num_visited = 0
        self.current_id = 0
        self.path = []  # Đường đi

    def hash_value(self):
        # Calculate the hash value S as described
        hash_val = self.current_id * (2 ** len(self.visited))
        for i in range(len(self.visited)):
            if self.visited[i]:
                hash_val += 2 ** i
        return hash_val

class StateWrapper:
    def __init__(self, cost, state):
        self.cost = cost
        self.state = state

    def __lt__(self, other):
        return self.cost < other.cost

def generate_random_tsp_instance(num_cities, seed):
    # Set the random seed for reproducibility
    random.seed(seed)

    # Generate random costs between 1 and 100, fill 0 when i == j
    costs = [[0 if i == j else random.randint(1, 100) for j in range(num_cities)] for i in range(num_cities)]

    return costs

def min_out_heuristic(state, costs):
    valid_cities = [i for i in range(len(costs)) if not state.visited[i]]
    if not valid_cities:
        return 0
    min_out_cost = min(costs[state.current_id][j] for j in valid_cities)

    return min_out_cost
    # return 0

def a_star(state, heuristic_func, costs, time_limit):
    start_time = timeit.default_timer()
    REACHED = [False] * (len(state.visited) * (2 ** len(state.visited)))
    priority_queue = [StateWrapper(0, state)]
    heapq.heapify(priority_queue)

    expanded_nodes = 0
    generated_nodes = 0

    while priority_queue:
        current_wrapper = heapq.heappop(priority_queue)
        current_cost, current_state = current_wrapper.cost, current_wrapper.state

        if timeit.default_timer() - start_time > time_limit:
            print("Time limit exceeded.")
            return None, None, None, None, None  # Time limit exceeded

        if REACHED[current_state.hash_value()] and REACHED[current_state.hash_value()] <= current_cost:
            continue

        REACHED[current_state.hash_value()] = current_cost
        expanded_nodes += 1

        if current_state.num_visited == len(costs) and current_state.current_id == 0 and current_state.visited[0]:
            end_time = timeit.default_timer()
            run_time = end_time - start_time
            current_state.path.append(current_state.current_id)  # Thêm thành phố cuối cùng vào đường đi
            print("Optimal Path:", current_state.path)
            return run_time, current_cost, expanded_nodes, generated_nodes, current_state

        for next_city in range(len(costs)):
            if not current_state.visited[next_city]:
                new_state = State(len(costs))
                new_state.visited = current_state.visited.copy()
                new_state.visited[next_city] = True
                new_state.num_visited = current_state.num_visited + 1
                new_state.current_id = next_city
                new_state.path = current_state.path + [current_state.current_id]  # Thêm thành phố vào đường đi mới

                # new_cost = current_cost + costs[current_state.current_id][next_city]
                new_cost = current_cost - min_out_heuristic(current_state, costs) + costs[current_state.current_id][next_city] + min_out_heuristic(new_state, costs)

                heapq.heappush(priority_queue, StateWrapper(new_cost, new_state))
                generated_nodes += 1    

    print("No solution found within the time limit.")
    return None, None, None, None, None

num_cities_list = [5, 10, 11, 12]
num_instances = 5
time_limit = 20 * 60  # 20 minutes in seconds

for num_cities in num_cities_list:
    for instance in range(1, num_instances + 1):
        seed = instance
        costs = generate_random_tsp_instance(num_cities, seed)
        initial_state = State(num_cities)

        run_time, optimal_cost, expanded_nodes, generated_nodes, _ = a_star(
            initial_state, min_out_heuristic, costs, time_limit
        )

        if run_time is not None and optimal_cost is not None:
            print(f"Results for {num_cities} cities (Seed {seed}):")
            print(f"Run Time: {run_time:.6f} seconds")
            print(f"Optimal Path Cost: {optimal_cost:.2f}")
            print(f"Number of Expanded Nodes: {expanded_nodes}")
            print(f"Number of Generated Nodes: {generated_nodes}")
            print()
