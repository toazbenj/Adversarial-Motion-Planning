from itertools import product


def generate_combinations(numbers, num_picks):
    """
    Generate all combinations of choices by picking `num_picks` times from the list `numbers`.

    Args:
        numbers (list): The list of numbers to pick from.
        num_picks (int): The number of times to pick.

    Returns:
        list of tuples: All combinations of length `num_picks`.
    """
    if not numbers or num_picks <= 0:
        return []

    # Use itertools.product to generate combinations
    combinations = list(product(numbers, repeat=num_picks))
    return combinations


# Example usage
numbers = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

num_picks = 3
result = generate_combinations(numbers, num_picks)

print("Input List:", numbers)
print(f"Combinations when picking {num_picks} times:")
for combo in result:
    print(combo)
