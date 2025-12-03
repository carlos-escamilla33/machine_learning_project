"""
D* Lite Algorithm Implementation

D* Lite is an incremental heuristic search algorithm for robot navigation
in unknown or partially known environments. It efficiently replans when
the environment changes without recomputing the entire path.

Based on: Koenig & Likhachev, "D* Lite" (2002)
"""

import heapq
import math
import random
from typing import List, Tuple, Set, Dict, Optional


class DStarLite:
    """
    D* Lite path planner for dynamic environments.

    Key differences from A*:
    1. Searches BACKWARDS from goal to start (so replanning is efficient)
    2. Uses two-key priority system for proper ordering
    3. Maintains rhs values (one-step lookahead) separate from g values
    4. Only updates affected nodes when environment changes
    """

    def __init__(self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]):
        """
        Initialize D* Lite planner.

        Args:
            grid: 2D grid where 0 = free space, 1 = obstacle
            start: Starting position (x, y) - where the robot is
            goal: Goal position (x, y) - where we want to go
        """
        self.grid = [row[:] for row in grid]  # Copy grid
        self.rows = len(grid)
        self.cols = len(grid[0])

        self.start = start  # Current robot position
        self.goal = goal

        # D* Lite specific data structures
        # g(s): estimate of distance from s to goal
        # rhs(s): one-step lookahead value (minimum of g(predecessor) + cost)
        self.g: Dict[Tuple[int, int], float] = {}
        self.rhs: Dict[Tuple[int, int], float] = {}

        # Priority queue with (key, node) pairs
        # Key is a tuple (k1, k2) for lexicographic ordering
        self.open_list: List[Tuple[Tuple[float, float], Tuple[int, int]]] = []
        self.open_set: Set[Tuple[int, int]] = set()

        # km: key modifier - accumulates as robot moves (handles changing start)
        self.km = 0.0

        # Initialize all nodes
        self._initialize()

    def _initialize(self):
        """Initialize the algorithm (corresponds to Initialize() in paper)"""
        # Set g and rhs to infinity for all nodes
        for i in range(self.rows):
            for j in range(self.cols):
                self.g[(i, j)] = float('inf')
                self.rhs[(i, j)] = float('inf')

        # Goal has rhs = 0 (we search backwards from goal)
        self.rhs[self.goal] = 0

        # Reset key modifier
        self.km = 0.0

        # Clear and initialize open list with goal
        self.open_list = []
        self.open_set = set()
        self._insert(self.goal)

    def heuristic(self, s1: Tuple[int, int], s2: Tuple[int, int]) -> float:
        """Heuristic function h(s1, s2) - Euclidean distance"""
        return math.sqrt((s1[0] - s2[0])**2 + (s1[1] - s2[1])**2)

    def _calculate_key(self, s: Tuple[int, int]) -> Tuple[float, float]:
        """
        Calculate priority key for node s.

        Key = (k1, k2) where:
        k1 = min(g(s), rhs(s)) + h(s_start, s) + km
        k2 = min(g(s), rhs(s))

        This ensures proper ordering during replanning.
        """
        g_val = self.g[s]
        rhs_val = self.rhs[s]
        min_val = min(g_val, rhs_val)

        k1 = min_val + self.heuristic(self.start, s) + self.km
        k2 = min_val

        return (k1, k2)

    def _insert(self, s: Tuple[int, int]):
        """Insert node into open list with calculated key"""
        if s not in self.open_set:
            key = self._calculate_key(s)
            heapq.heappush(self.open_list, (key, s))
            self.open_set.add(s)

    def _remove(self, s: Tuple[int, int]):
        """Mark node as removed from open list"""
        if s in self.open_set:
            self.open_set.remove(s)

    def _update(self, s: Tuple[int, int]):
        """Update node in open list (remove and re-insert with new key)"""
        self._remove(s)
        self._insert(s)

    def _top_key(self) -> Tuple[float, float]:
        """Get the minimum key from open list"""
        # Clean up stale entries
        while self.open_list and self.open_list[0][1] not in self.open_set:
            heapq.heappop(self.open_list)

        if self.open_list:
            return self.open_list[0][0]
        return (float('inf'), float('inf'))

    def _pop(self) -> Optional[Tuple[int, int]]:
        """Pop node with minimum key from open list"""
        while self.open_list:
            key, s = heapq.heappop(self.open_list)
            if s in self.open_set:
                self.open_set.remove(s)
                return s
        return None

    def _key_less_than(self, k1: Tuple[float, float], k2: Tuple[float, float]) -> bool:
        """Lexicographic comparison of keys"""
        return k1[0] < k2[0] or (k1[0] == k2[0] and k1[1] < k2[1])

    def get_neighbors(self, s: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring cells (8-connected grid)"""
        neighbors = []
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        for dx, dy in directions:
            nx, ny = s[0] + dx, s[1] + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                neighbors.append((nx, ny))

        return neighbors

    def cost(self, s1: Tuple[int, int], s2: Tuple[int, int]) -> float:
        """
        Movement cost from s1 to s2.
        Returns infinity if either cell is an obstacle.
        """
        # Check if either cell is an obstacle
        if self.grid[s1[0]][s1[1]] == 1 or self.grid[s2[0]][s2[1]] == 1:
            return float('inf')

        # Diagonal vs straight movement cost
        dx = abs(s1[0] - s2[0])
        dy = abs(s1[1] - s2[1])

        if dx + dy == 2:  # Diagonal
            return math.sqrt(2)
        return 1.0

    def _update_vertex(self, u: Tuple[int, int]):
        """
        Update vertex u (corresponds to UpdateVertex() in paper).

        This is the core of D* Lite:
        - If u != goal: rhs(u) = min over successors s' of (c(u,s') + g(s'))
        - If g(u) != rhs(u): u is inconsistent, add to open list
        - Else: u is consistent, remove from open list
        """
        if u != self.goal:
            # Calculate rhs as minimum over all successors
            min_rhs = float('inf')
            for s_prime in self.get_neighbors(u):
                candidate = self.cost(u, s_prime) + self.g[s_prime]
                if candidate < min_rhs:
                    min_rhs = candidate
            self.rhs[u] = min_rhs

        # Remove from open list if present
        self._remove(u)

        # If inconsistent (g != rhs), add to open list
        if self.g[u] != self.rhs[u]:
            self._insert(u)

    def compute_shortest_path(self) -> bool:
        """
        Main planning loop (corresponds to ComputeShortestPath() in paper).

        Processes nodes until start is consistent and has correct g-value.
        Returns True if path found, False otherwise.
        """
        iterations = 0
        max_iterations = self.rows * self.cols * 2  # Safety limit

        while iterations < max_iterations:
            # Check termination conditions
            top_key = self._top_key()
            start_key = self._calculate_key(self.start)

            # Terminate if open list is empty or start is consistent with correct key
            if not self._key_less_than(top_key, start_key) and self.rhs[self.start] == self.g[self.start]:
                break

            # Get node with minimum key
            u = self._pop()
            if u is None:
                break

            k_old = top_key
            k_new = self._calculate_key(u)

            if self._key_less_than(k_old, k_new):
                # Key has increased, re-insert with new key
                self._insert(u)
            elif self.g[u] > self.rhs[u]:
                # Overconsistent: make consistent
                self.g[u] = self.rhs[u]
                # Update all predecessors
                for s in self.get_neighbors(u):
                    self._update_vertex(s)
            else:
                # Underconsistent: make overconsistent then update
                self.g[u] = float('inf')
                # Update u and all predecessors
                self._update_vertex(u)
                for s in self.get_neighbors(u):
                    self._update_vertex(s)

            iterations += 1

        # Check if path exists
        return self.g[self.start] != float('inf')

    def plan(self) -> List[Tuple[int, int]]:
        """
        Compute initial path from start to goal.
        Returns: Path as list of (x, y) coordinates, empty if no path exists.
        """
        if self.compute_shortest_path():
            return self.extract_path()
        return []

    def extract_path(self) -> List[Tuple[int, int]]:
        """
        Extract path by greedily following minimum cost successors.
        D* Lite stores costs TO goal, so we follow decreasing g-values.
        """
        if self.g[self.start] == float('inf'):
            return []

        path = [self.start]
        current = self.start

        max_steps = self.rows * self.cols  # Safety limit
        steps = 0

        while current != self.goal and steps < max_steps:
            # Find successor with minimum (cost + g)
            best_next = None
            best_cost = float('inf')

            for s_prime in self.get_neighbors(current):
                candidate_cost = self.cost(current, s_prime) + self.g[s_prime]
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_next = s_prime

            if best_next is None or best_cost == float('inf'):
                return []  # No valid path

            path.append(best_next)
            current = best_next
            steps += 1

        return path if current == self.goal else []

    def update_start(self, new_start: Tuple[int, int]):
        """
        Update robot position (when robot moves along path).
        Adjusts km to maintain correct key values.
        """
        # km += h(s_last, s_start)
        self.km += self.heuristic(self.start, new_start)
        self.start = new_start

    def update_obstacles(self, changed_cells: List[Tuple[Tuple[int, int], int]]) -> List[Tuple[int, int]]:
        """
        Handle environment changes and replan.

        This is where D* Lite shines - it only updates affected nodes,
        not the entire search space.

        Args:
            changed_cells: List of ((x, y), new_value) where new_value is 0 or 1

        Returns: New path after replanning
        """
        if not changed_cells:
            return self.extract_path()

        # Update grid and affected vertices
        for (x, y), new_val in changed_cells:
            old_val = self.grid[x][y]
            if old_val == new_val:
                continue

            # Update grid
            self.grid[x][y] = new_val

            # Update all edges to/from this cell
            # The cell itself and all its neighbors are affected
            self._update_vertex((x, y))
            for neighbor in self.get_neighbors((x, y)):
                self._update_vertex(neighbor)

        # Recompute shortest path (efficiently - only processes affected nodes)
        if self.compute_shortest_path():
            return self.extract_path()
        return []

    def add_obstacles(self, obstacles: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Convenience method to add new obstacles.

        Args:
            obstacles: List of (x, y) positions to become obstacles

        Returns: New path after replanning
        """
        changed = [((x, y), 1) for x, y in obstacles
                   if self.grid[x][y] == 0]  # Only actually changed cells
        return self.update_obstacles(changed)

    def remove_obstacles(self, cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Convenience method to remove obstacles (make cells free).

        Args:
            cells: List of (x, y) positions to become free

        Returns: New path after replanning
        """
        changed = [((x, y), 0) for x, y in cells
                   if self.grid[x][y] == 1]  # Only actually changed cells
        return self.update_obstacles(changed)

    def create_random_obstacles(self, count: int = 6) -> List[Tuple[int, int]]:
        """Generate random obstacle positions (avoiding start and goal)"""
        obstacles = []
        attempts = 0
        max_attempts = count * 10

        while len(obstacles) < count and attempts < max_attempts:
            x = random.randint(0, self.rows - 1)
            y = random.randint(0, self.cols - 1)

            if ((x, y) != self.start and
                (x, y) != self.goal and
                self.grid[x][y] == 0 and
                    (x, y) not in obstacles):
                obstacles.append((x, y))

            attempts += 1

        return obstacles


def visualize_grid(grid: List[List[int]], path: List[Tuple[int, int]],
                   start: Tuple[int, int], goal: Tuple[int, int]):
    """Visualize grid with path"""
    visual = [row[:] for row in grid]

    # Mark path
    for x, y in path:
        if visual[x][y] == 0:
            visual[x][y] = 2

    # Mark start and goal
    if start:
        visual[start[0]][start[1]] = 3
    if goal:
        visual[goal[0]][goal[1]] = 4

    symbols = {0: '·', 1: '█', 2: '*', 3: 'S', 4: 'G'}
    print("\nGrid visualization:")
    print("(· = free, █ = obstacle, * = path, S = start, G = goal)")
    for row in visual:
        print(' '.join(symbols.get(cell, '?') for cell in row))
    print()


def main():
    """Demonstrate D* Lite algorithm"""

    # Create a larger grid for better demonstration
    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]

    start = (0, 7)
    goal = (7, 0)

    print("=" * 50)
    print("D* Lite Path Planning Demo")
    print("=" * 50)

    # Create planner
    planner = DStarLite(grid, start, goal)

    # Compute initial path
    print("\n1. Computing initial path...")
    path = planner.plan()

    if path:
        print(f"   Path found with {len(path)} steps")
        print(f"   Path: {path}")
        visualize_grid(grid, path, start, goal)
    else:
        print("   No path found!")
        return

    # Simulate discovering obstacles
    print("\n2. Discovering new obstacles...")
    new_obstacles = planner.create_random_obstacles(8)
    print(f"   New obstacles at: {new_obstacles}")

    # Replan with new obstacles - this is where D* Lite efficiency shows
    print("\n3. Replanning (D* Lite efficiently updates only affected nodes)...")
    new_path = planner.add_obstacles(new_obstacles)

    if new_path:
        print(f"   New path found with {len(new_path)} steps")
        print(f"   Path: {new_path}")
        visualize_grid(planner.grid, new_path, start, goal)
    else:
        print("   No path exists with new obstacles!")
        visualize_grid(planner.grid, [], start, goal)

    # Demonstrate obstacle removal
    if new_obstacles and new_path:
        print("\n4. Removing some obstacles...")
        to_remove = new_obstacles[:3]
        print(f"   Removing obstacles at: {to_remove}")

        restored_path = planner.remove_obstacles(to_remove)

        if restored_path:
            print(f"   Path after removal: {len(restored_path)} steps")
            visualize_grid(planner.grid, restored_path, start, goal)

    # Demonstrate moving start position
    print("\n5. Simulating robot movement...")
    if new_path and len(new_path) > 2:
        # Robot moves along path
        new_start = new_path[2]  # Move to third position
        print(f"   Robot moved from {start} to {new_start}")
        planner.update_start(new_start)

        moved_path = planner.extract_path()
        if moved_path:
            print(f"   Remaining path: {len(moved_path)} steps")
            visualize_grid(planner.grid, moved_path, new_start, goal)


if __name__ == "__main__":
    main()
