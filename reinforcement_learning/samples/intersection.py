def intersect(line1, line2):
    """
    Checks if two line segments intersect.

    Args:
        line1 (list): A list of two tuples representing the endpoints of the first line segment.
        line2 (list): A list of two tuples representing the endpoints of the second line segment.

    Returns:
        bool: True if the line segments intersect, False otherwise.
    """

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    A, B = line1
    C, D = line2

    is_same_point = A == C or B == D or A == D or B == C

    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D) or is_same_point

line1 = [(1, 1), (4, 4)]
line2 = [(1,1), (4, 3)]

if intersect(line1, line2):
    print("Lines intersect.")
else:
    print("Lines do not intersect.")