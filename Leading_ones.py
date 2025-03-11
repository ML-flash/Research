class LeadingOnesFitness:
    def __init__(self, max_length, update_best_func, phase_switch_generations=50):
        """
        Initialize the LeadingOnesFitness class.
        
        Args:
            max_length: Maximum length to evaluate in the solution
            update_best_func: Function to update the best organism
            phase_switch_generations: Number of generations after which to switch phases
        """
        self.max_length = max_length
        self.update_best = update_best_func  # Store the function for updating the best organism
        self.genes = ['0', '1']  # Allowed genes
        self.target_gene = '1'  # Start by optimizing for leading 1s
        self.phase = 1  # Tracks which phase we're in (1: maximizing '1's, 2: maximizing '0's)
        self.phase_switch_generations = phase_switch_generations
        self.last_phase_switch = 0  # Track when we last switched phases

    def compute(self, encoded_individual, ga_instance):
        # Check if we need to switch phases based on generation count
        current_generation = ga_instance.current_generation
        
        if (current_generation - self.last_phase_switch) >= self.phase_switch_generations:
            self.phase = 2 if self.phase == 1 else 1  # Toggle between phases
            self.target_gene = '0' if self.phase == 2 else '1'  # Update target
            self.last_phase_switch = current_generation
            
            # Output information about phase switch
            print(f"Generation {current_generation}: Switching to phase {self.phase}, targeting '{self.target_gene}'")
            print("Resetting best organism tracking for new phase")
            
            # Reset best organism tracking as before
            self.reset_best_organism = True
            
            # We also need to modify the update_best_func behavior temporarily
            # Store the original function
            self._original_update_best = self.update_best
            
            # Replace with a version that updates the runner's best_organism dict
            def reset_update_best(genome, fitness, verbose=True):
                # Call the original function, which will handle the experiment runner's dictionary
                self._original_update_best(genome, fitness, verbose=verbose)
                # Reset our flag once it's been used
                self.reset_best_organism = False
                # Restore the original function
                self.update_best = self._original_update_best
                
            # Set the update function to our temporary version
            self.update_best = reset_update_best
            
        # Decode the individual
        decoded_individual = ga_instance.decode_organism(encoded_individual)

        # Initialize fitness score and count of consecutive target genes
        fitness_score = 0
        target_count = 0

        # Count leading target genes (either '1' or '0' depending on phase)
        for i, gene in enumerate(decoded_individual):
            if i >= self.max_length:
                break  # Stop counting at max_length

            if gene == self.target_gene:
                fitness_score += 1
                target_count += 1
            else:
                break  # Stop at first mismatch

        # Apply penalty for exceeding target length
        if len(decoded_individual) > self.max_length:
            # Penalty of 0.5 points per extra gene
            extra_length = len(decoded_individual) - self.max_length
            fitness_score -= 0.5 * extra_length

        # Update the best organism using the passed function
        self.update_best(encoded_individual, fitness_score, verbose=True)

        # Return the final fitness score
        return fitness_score
