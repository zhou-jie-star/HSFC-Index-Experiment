import numpy as np


def calc_binary(x):
    """Convert integer to binary array"""
    if isinstance(x, (int, np.integer)):
        binary_str = bin(x)[2:]
        return np.array([int(bit) for bit in binary_str], dtype=np.int8)
    return np.array([], dtype=np.int8)


def get_theta(bit_array):
    """Generate weight array"""
    # Simple error handling
    if len(bit_array) == 0:
        return np.array([], dtype=np.int64)
    theta = np.full(len(bit_array), 2, dtype=np.int64) ** bit_array
    return theta


def calc_sfc_position_multidim(coordinates, theta):
    """Calculate SFC value for multi-dimensional coordinates"""
    coordinates = np.array(coordinates)
    n_dims = len(coordinates)
    sfc_position = 0

    for dim in range(n_dims):
        if dim >= len(theta): break

        coord_val = int(coordinates[dim])
        # Get binary representation and reverse (LSB first) to align with weights
        coord_binary = calc_binary(coord_val)[::-1]

        theta_dim = theta[dim]
        # Ensure length matching
        length = min(len(coord_binary), len(theta_dim))

        if length > 0:
            dim_contribution = np.sum(coord_binary[:length] * theta_dim[:length])
            sfc_position += dim_contribution

    return sfc_position


def create_sfc_multidim(data_points, theta):
    """Create and sort SFC curve"""
    curve = []
    for point in data_points:
        point = np.array(point)
        try:
            sfc_val = calc_sfc_position_multidim(point, theta)
            # Store (SFC value, coordinates...)
            # Note: Convert numpy types to native types to prevent errors in older numpy versions
            entry = tuple([float(sfc_val)] + point.tolist())
            curve.append(entry)
        except Exception:
            continue

    # Sort by SFC value
    curve.sort(key=lambda x: x[0])
    return curve


def get_dimension_count(data_points):
    """Get data dimension count"""
    if not data_points: return 0
    return len(np.array(data_points[0]))


def validate_multidim_compatibility(points, queries):
    """Validate data and query dimension consistency"""
    if not points: raise ValueError("No points provided")
    dim = len(points[0])
    if queries:
        q = queries[0]
        # Compatible with different query object structures
        if hasattr(q, 'min_point'):
            q_dim = len(q.min_point)
        elif isinstance(q, (list, tuple)) and len(q) > 0:
            q_dim = len(q[0])
        else:
            q_dim = dim  # Skip check if unable to determine

        if q_dim != dim:
            raise ValueError(f"Dimension mismatch: Points are {dim}D, but Queries appear to be {q_dim}D")


# --- Compatibility patch functions (Fix Import Errors) ---

def merge_sort_with_custom_order(arr, bit_length=None):
    """
    Compatibility patch: Used for sfc_optimization.py import.
    In simplified version, directly use numpy argsort.
    """
    return np.argsort(arr)


def merge_sort_with_custom_order_multidim(arr, n_dimensions=None, bit_length=None):
    """
    Compatibility patch: Multi-dimensional sorting helper function.
    """
    return np.argsort(arr)


def generate_multidim_theta_structure(n_dimensions, bit_length):
    """
    Generate initial theta structure (for optimizer initialization).
    Returns a list where each element is the initial weight array for the corresponding dimension.
    """
    structure = []
    for _ in range(n_dimensions):
        # Default weights: 2^0, 2^1, ... 2^(bit_length-1)
        # This is just initialization; specific values will be overwritten by Bayesian optimization
        weights = np.array([i for i in range(bit_length)], dtype=np.int64)
        structure.append(weights)
    return structure