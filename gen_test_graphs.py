import random
import os

def create_graph(filename, num_vertices, num_edges):
    """Create a random directed graph"""
    os.makedirs('graphs', exist_ok=True)
    
    with open(f'graphs/{filename}', 'w') as f:
        f.write(f'{num_vertices} {num_edges}\n')
        
        # Generate random edges
        edges = set()
        while len(edges) < num_edges:
            u = random.randint(0, num_vertices - 1)
            v = random.randint(0, num_vertices - 1)
            if u != v:
                edges.add((u, v))
        
        for u, v in edges:
            weight = random.randint(-50, 100)
            f.write(f'{u} {v} {weight}\n')
    
    print(f'[OK] Created {filename}: {num_vertices} vertices, {num_edges} edges')

# Create test graphs of various sizes
create_graph('tiny.txt', 5, 8)
create_graph('small.txt', 20, 50)
create_graph('medium.txt', 50, 150)
create_graph('large.txt', 100, 500)

print('\n[OK] Test graphs created successfully!')
