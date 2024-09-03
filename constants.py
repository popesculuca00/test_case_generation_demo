import re
from collections import defaultdict
import pynvml

PROMPT_TEMPLATE = """
### Input:
{}

### Response:
{}"""


EMPTY_TEST_MSG = "# You need to generate the tests"

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a speechless code assistant. A user will give you a code representing a python file. 
Your task is to take said code and generate a complete working testing file using Pytest. 
You may assume the original code can be found in a `source.py` file residing in the same directory as the test file, you must import it as such.
You *MUST* always provide the fully updated code with no other explanations.

### Input:
{}

### Response:
{}
"""


class CustomDefaultDict(defaultdict):
    def __init__(self, default_factory=None, *args, **kwargs):
        super().__init__(default_factory, *args, **kwargs)

    def keys(self):
        return [key for key in super().keys() if key is not None]

    def items(self):
        return [(key, value) for key, value in super().items() if key is not None]

    def __iter__(self):
        return iter(self.keys())
    



code_snippets = CustomDefaultDict(lambda : "")
code_snippets.update({
    "Find missing number": "def find_missing_number(numbers):\n    n = len(numbers) + 1\n    expected_sum = n * (n + 1) // 2\n    actual_sum = sum(numbers)\n    return {'result': expected_sum - actual_sum}",
    "Degrees": """def dms(degrees):\n    degrees_int = int(abs(degrees))\t # integer degrees\n    degrees_frac = abs(degrees) - degrees_int  # fractional degrees, used to compute minutes\n    minutes_int = float(int(degrees_frac * 60))  # integer minutes\n    minutes_frac = degrees_frac - minutes_int / 60  # fractional minutes, used to compute seconds\n    seconds = minutes_frac * 3600  # decimal seconds\n\n    # Handle sign.  Degrees portion will contain the sign of the coordinate.\n    # Minutes and seconds will always be positive.\n    # sign function returns -1, 0, +1 for x < 0, x == 0, x > 0, respectively\n    if degrees < 0:\n        degrees_int *= -1\n\n    return degrees_int, minutes_int, seconds""",
    "Transform grad": """def transform_grad_batch_min_max(batch_grad):\n\n    batch_size = batch_grad.shape[0]\n    return [\n        batch_size * batch_grad.data.min().item(),\n        batch_size * batch_grad.data.max().item(),\n    ]""",
    "Ratios to coordinates": """def ratios_to_coordinates(bx, by, bw, bh, width, height):\n    \n    w, h = bw * width, bh * height\n    x, y = bx * width + (w / 2), by * height + (h / 2)\n    return x, y, x + w, y + h""",
    "Critical damping": """def critical_damping_parameters(theta, order=2):\r\n    \r\n    if theta < 0 or theta > 1:\r\n        raise ValueError('theta must be between 0 and 1')\r\n\r\n    if order == 2:\r\n        return (1. - theta**2, (1. - theta)**2)\r\n\r\n    if order == 3:\r\n        return (1. - theta**3, 1.5*(1.-theta**2)*(1.-theta), .5*(1 - theta)**3)\r\n\r\n    raise ValueError('bad order specified: {}'.format(order))""",
    "Set size": """def set_size(width, fraction=1):\r\n    \r\n    # Width of figure (in pts)\r\n    fig_width_pt = width * fraction\r\n\r\n    # Convert from pt to inches\r\n    inches_per_pt = 1 / 72.27\r\n\r\n    # Golden ratio to set aesthetic figure height\r\n    # https://disq.us/p/2940ij3\r\n    golden_ratio = (5**.5 - 1) / 2\r\n\r\n    # Figure width in inches\r\n    fig_width_in = fig_width_pt * inches_per_pt\r\n    # Figure height in inches\r\n    fig_height_in = fig_width_in * golden_ratio\r\n\r\n    fig_dim = (fig_width_in, fig_height_in)\r\n\r\n    return fig_dim""",
    "White percentage": """def white_percentage(rgb, white_thresh=220, tolerance=0.8):\r\n    \r\n    score = (rgb.mean(axis=-1) > white_thresh).sum() # amount of white pixels\r\n    score = score / (rgb.shape[0] * rgb.shape[1])\r\n    accepted = score < tolerance\r\n    return accepted""",
    "I to white": """def i_to_white(i, normalize=False):\r\n    \r\n    i = max(i, 0.0)\r\n    i = min(i, 1.0)\r\n    rgb = min(i * 255, 255)\r\n    if not normalize:\r\n        return int(rgb), int(rgb), int(rgb)\r\n    rgb = rgb / 255\r\n    return rgb, rgb, rgb""",
    "Thrust correction": """def double_ion_thrust_correction(double_fraction):\r\n    \r\n    if double_fraction < 0 or double_fraction > 1:\r\n        raise ValueError('double_fraction {:.f} is not in [0, 1]'.format(double_fraction))\r\n\r\n    return (1  + (0.5)**0.5 * double_fraction) / (1 + double_fraction)""",
    "Abs smooth dv": """def abs_smooth_dv(x, x_deriv, delta_x):\r\n    \r\n    if x >= delta_x:\r\n        y_deriv = x_deriv\r\n        y = x\r\n\r\n    elif x <= -delta_x:\r\n        y_deriv = -x_deriv\r\n        y = -x\r\n\r\n    else:\r\n        y_deriv = 2.0 * x * x_deriv / (2.0 * delta_x)\r\n        y = x**2 / (2.0 * delta_x) + delta_x / 2.0\r\n\r\n    return y, y_deriv""",
    "Degree": """def Degree(adjmatrix, directed=False):\r\n    \r\n    N = len(adjmatrix)\r\n    adjmatrix = adjmatrix.astype('bool')\r\n\r\n    if directed:\r\n        indegree = adjmatrix.sum(axis=0)\r\n        outdegree = adjmatrix.sum(axis=1)\r\n        return indegree, outdegree\r\n\r\n    else:\r\n        degree = adjmatrix.sum(axis=1)\r\n        return degree""",
    "Saturation correlate": """def saturation_correlate(M, Q):\r\n    \r\n\r\n    s = 100 * (M / Q) ** 0.5\r\n    return s""",
    "Compute ring radius": """def compute_ring_radius_from_master_curve(l, dp=160., do=25.6, N=8, lowerGamma=2.2, setting='medium'):\r\n    \r\n    if setting == 'medium':\r\n        dp, do, N = 160, 25.6, 8.  # Setting 1\r\n    elif setting == 'small':  # Setting 2\r\n        dp, do, N = 56.7, 12.8, 8.\r\n    lstar = 1. / N * (dp / do) ** 2 * l / do\r\n    radius = do / lowerGamma * lstar ** (1 / 3)\r\n    return radius""",
    "Max occurences": "def max_occurrences(list1):\n    max_val = 0\n    result = list1[0] \n    for i in list1:\n        occu = list1.count(i)\n        if occu > max_val:\n            max_val = occu\n            result = i \n    return [result]",
    "Is date in range": "import datetime\n\ndef is_date_within_range(date, start_date, end_date):\n    if start_date <= date <= end_date:\n        return True\n    else:\n        return False",
    "Guess Dimensions": """def guess_spatial_dimensions(image):\n    \n    if image.ndim == 2:\n        return 2\n    if image.ndim == 3 and image.shape[-1] != 3:\n        return 3\n    if image.ndim == 3 and image.shape[-1] == 3:\n        return None\n    if image.ndim == 4 and image.shape[-1] == 3:\n        return 3\n    else:\n        raise ValueError("Expected 2D, 3D, or 4D array, got %iD." % image.ndim)""",
    "Compute pad": """def compute_pad(image_shape, kernel_size, enforce_odd=True):\r\n  \r\n  padding = (kernel_size//2, kernel_size//2)\r\n  if enforce_odd:\r\n    adjust = (1 - image_shape[0] % 2, 1 - image_shape[1] % 2)\r\n  else:\r\n    adjust = (0, 0)\r\n  return ((padding[0] - adjust[0], padding[0]), (padding[1] - adjust[1],\r\n                                                 padding[1]))""",
    "Conv output length": """def conv_output_length(input_length, filter_size, stride, pad=0):\r\n    \r\n    if input_length is None:\r\n        return None\r\n    if pad == 'valid':\r\n        output_length = input_length - filter_size + 1\r\n    elif pad == 'full':\r\n        output_length = input_length + filter_size - 1\r\n    elif pad == 'same':\r\n        output_length = input_length\r\n    elif isinstance(pad, int):\r\n        output_length = input_length + 2 * pad - filter_size + 1\r\n    else:\r\n        raise ValueError('Invalid pad: {0}'.format(pad))\r\n\r\n    # This is the integer arithmetic equivalent to\r\n    # np.ceil(output_length / stride)\r\n    output_length = (output_length + stride - 1) // stride\r\n\r\n    return output_length""",
    "Is inside": """def inside(x, y, area):\r\n    \r\n    x1, x2, y1, y2 = area\r\n    return ((x >= x1) & (x <= x2) & (y >= y1) & (y <= y2))""",
    "Filter results": """def filter_result(results, scores, score_thr):\r\n    \r\n    assert results.ndim == 2\r\n    assert scores.shape[0] == results.shape[0]\r\n    assert isinstance(score_thr, float)\r\n    assert 0 <= score_thr <= 1\r\n\r\n    inds = scores > score_thr\r\n    valid_results = results[inds, :]\r\n    valid_scores = scores[inds]\r\n    return valid_results, valid_scores""",
    "Dice loss": """def dice_loss(inputs, targets, num_boxes):\r\n    \r\n    inputs = inputs.sigmoid()\r\n    inputs = inputs.flatten(1)\r\n    targets = targets.flatten(1)\r\n\r\n    numerator = 2 * (inputs * targets).sum(1)\r\n    denominator = inputs.sum(-1) + targets.sum(-1)\r\n    loss = 1 - (numerator + 1) / (denominator + 1)\r\n    return loss.sum() / num_boxes""",
    "Stepsize": """def stepsize(n_levels):\r\n    \r\n    assert float(\r\n        n_levels / 2\r\n    ).is_integer(), "n_levels must be an even number, see function docstring."\r\n\r\n    step = n_levels / (2 * (n_levels - 1))\r\n\r\n    return step""",
    "Compute resize scale": """def compute_resize_scale(image_shape, min_side=800, max_side=1333):\r\n    \r\n    (rows, cols, _) = image_shape\r\n\r\n    smallest_side = min(rows, cols)\r\n    print(smallest_side)\r\n\r\n    # rescale the image so the smallest side is min_side\r\n    scale = min_side / smallest_side\r\n\r\n    # check if the largest side is now greater than max_side, which can happen\r\n    # when images have a large aspect ratio\r\n    largest_side = max(rows, cols)\r\n    if largest_side * scale > max_side:\r\n        scale = max_side / largest_side\r\n\r\n    return scale""",
    "Standard to flattened": """def standard_to_flattened(positions):\r\n    \r\n    n_atoms = positions.shape[-2]\r\n    if len(positions.shape) > 2:\r\n        batch_size = positions.shape[0]\r\n        flattened_shape = (batch_size, n_atoms*3)\r\n    else:\r\n        flattened_shape = (n_atoms*3,)\r\n    return positions.reshape(flattened_shape)""",
    "Pearson multi": """def pearson_multi(X):\r\n    \r\n    from scipy.special import betainc\r\n    from sklearn.covariance import EmpiricalCovariance\r\n    from sklearn.preprocessing import StandardScaler\r\n\r\n    X = StandardScaler().fit_transform(X)\r\n\r\n    corr = EmpiricalCovariance(\r\n        store_precision=False,\r\n        assume_centered=True).fit(X).covariance_\r\n    corr[corr > 1.] = 1.\r\n    corr[corr < -1.] = -1.\r\n\r\n    return corr""",
    "Str to orientation": """def str_to_orientation(value, reversed_horizontal=False, reversed_vertical=False):\r\n    \r\n\r\n    aliases = {"left-right": "lr", "right-left": "rl", "top-bottom": "tb",\r\n            "bottom-top": "bt", "top-down": "tb", "bottom-up": "bt",\r\n            "top-bottom": "tb", "bottom-top": "bt", "td": "tb", "bu": "bt"}\r\n\r\n    dir = ["lr", "rl"][reversed_horizontal]\r\n    aliases.update(horizontal=dir, horiz=dir, h=dir)\r\n\r\n    dir = ["tb", "bt"][reversed_vertical]\r\n    aliases.update(vertical=dir, vert=dir, v=dir)\r\n\r\n    result = aliases.get(value, value)\r\n    if result not in ("lr", "rl", "tb", "bt"):\r\n        raise ValueError("unknown orientation: %s" % result)\r\n    return result""",
    "Clamp value": """def clamp_value(value, minimum, maximum):\r\n    \r\n    if value < minimum:\r\n        return minimum\r\n    elif value > maximum:\r\n        return maximum\r\n    else:\r\n        return value""",
    "Epoch scheduler": """def lambda_lr_schedule(epoch):\r\n    \r\n    lr_scale = 1.\r\n    if epoch > 180:\r\n        lr_scale = 0.5e-3\r\n    elif epoch > 160:\r\n        lr_scale = 1e-3\r\n    elif epoch > 120:\r\n        lr_scale = 1e-2\r\n    elif epoch > 80:\r\n        lr_scale = 1e-1\r\n    return lr_scale""",
    "Get position leftmost digit": """def get_position_of_leftmost_digit(number):\r\n    \r\n\r\n    if number == 0:\r\n        return None\r\n    number = abs(number)\r\n\r\n    pos = 0\r\n    while (number - number % (10 ** -pos)) != 0:\r\n        pos -= 1\r\n    while (number - number % (10 ** -pos)) == 0:\r\n        pos += 1\r\n    return pos""",
    "Kinetic energy": """import torch\r\n\r\ndef kin_energy(momentum, inv_mass):\r\n    \r\n    if isinstance(inv_mass, torch.Tensor):\r\n        if inv_mass.numel() == momentum.numel(): # Assuming diagonal mass\r\n            ke = .5 * torch.dot(momentum, inv_mass * momentum)\r\n        else: # Assuming full matrix\r\n            ke = .5 *  torch.dot(momentum, torch.matmul(inv_mass, momentum))\r\n    else: # Assuming a single scalar\r\n        ke = .5 * inv_mass * torch.dot(momentum, momentum)\r\n\r\n    return ke""",
    "Custom": ""
})


available_models = CustomDefaultDict(lambda: "")
available_models.update({
    "Fine-Tuned": "outputs/checkpoint-7000",
    "Llama 3.1 8b": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "Phi-3": "unsloth/Phi-3-mini-4k-instruct"
})

def extract_code(response):
    pattern = r'```(?:python)?\n?(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    
    if matches:
        return '\n\n'.join(match.strip() for match in matches)
    else:
        return response.strip()
    

def get_gpu_memory_info():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total = mem_info.total / 1024**2
    used = mem_info.used / 1024**2
    free = mem_info.free / 1024**2
    pynvml.nvmlShutdown()
    return total, used, free