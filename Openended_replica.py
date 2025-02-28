class PathTrajectoryFitness:
    def __init__(self, visit_counts_dict, update_best_func):
        """
        :param visit_counts_dict: A dictionary tracking total visits for each (x,y) location,
                                  accumulated over the entire run. E.g. {(0, 0): 10, (1, 0): 5, ...}
        :param update_best_func:  A callback that accepts (encoded_individual, fitness_score, verbose)
                                  and updates the 'best so far' if the new score is higher.
        """
        self.visit_counts = visit_counts_dict
        self.update_best = update_best_func
        self.genes = ['left', 'right', 'forward']

    def compute(self, encoded_individual, ga_instance):
        """
        Compute the fitness of a single genome by:
          1) Decoding to get ['left','right','forward',...].
          2) Starting at (0,0) facing 'up' = (dx, dy) = (0, 1).
          3) Rotating or moving according to each gene.
          4) Summing fitness = ∑(1 / (visit_counts[loc] + 1)) for each unique visited location.
          5) Incrementing self.visit_counts[loc] after awarding points.

        :param encoded_individual: The organism in encoded (integer) form.
        :param ga_instance:       Reference to the GA, used here to decode the genome.
        :return:                  The numeric fitness.
        """
        # 1) Decode the organism to get the actual instructions
        decoded_genome = ga_instance.decode_organism(encoded_individual)

        # 2) Initialize position & heading
        x, y = 0, 0           # start at origin
        dx, dy = 0, 1         # facing "up" along the y-axis
        visited_positions = set()
        visited_positions.add((x, y))  # we count the starting cell as visited

        # Helper functions for rotation
        def turn_left(d_x, d_y):
            # (up -> left -> down -> right -> up)
            return (-d_y, d_x)

        def turn_right(d_x, d_y):
            # (up -> right -> down -> left -> up)
            return (d_y, -d_x)

        # 3) Walk the path
        for gene in decoded_genome:
            if gene == 'left':
                dx, dy = turn_left(dx, dy)
            elif gene == 'right':
                dx, dy = turn_right(dx, dy)
            elif gene == 'forward':
                x += dx
                y += dy
            # else: ignore unrecognized genes, if any
            visited_positions.add((x, y))

        # 4) Compute fitness based on how often each location was previously visited
        fitness_score = 0.0
        for loc in visited_positions:
            old_visits = self.visit_counts.get(loc, 0)
            fitness_score += 1.0 / (old_visits + 1)

        # 5) Update the global visit counts so future organisms see more visits for these locations
        for loc in visited_positions:
            self.visit_counts[loc] = self.visit_counts.get(loc, 0) + 1

        # Update best-organism tracking
        self.update_best(encoded_individual, fitness_score, verbose=True)

        return fitness_score
