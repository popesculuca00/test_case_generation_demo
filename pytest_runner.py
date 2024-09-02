import string
import random
import sys, subprocess, os, shutil
import re

from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from functools import wraps

def timeout(time_limit):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=time_limit)
                except TimeoutError:
                    executor.shutdown(wait=False)
                    raise TimeoutError(f"{func.__name__} exceeded timeout of {time_limit} seconds")
        return wrapper
    return decorator



def strip_ansi_escape_sequences(s):
    ansi_escape_regex = re.compile(r"\x1b\[([0-9A-Za-z]+)(;[0-9]+)*m")
    return ansi_escape_regex.sub("", s)


def get_num_fails(result: subprocess.CompletedProcess[str]) -> int:
    """
    Extracts and parses the number of fails a testing run has from the pytest output.
    """
    num_fails = re.search("(\d+) failed", result.stdout)
    if num_fails is None:
        num_fails = 0
    else:
        num_fails = int(num_fails[1])
    return num_fails


def get_coverage_percent(result: subprocess.CompletedProcess[str]) -> float:
    """
    Extracts and parses the test coverage percentage from the pytest output.
    """
    coverage_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", result.stdout)
    if coverage_match:
        coverage_percent = coverage_match.group(1)
    else:
        coverage_percent = 0.0
    try:
        coverage_percent = float(coverage_percent)
    except ValueError as e:
        print(f"Error converting {coverage_percent} to float")
        coverage_percent = 0
    return coverage_percent


def get_test_case_fails(result: subprocess.CompletedProcess[str]) -> str:
    """
    Extracts all specific fails from the pytest output under the `short test summary info` section.
    """
    # test_cases = re.findall("(FAILED.*)$", result.stdout)
    test_cases = re.findall("(FAILED\s*test_source.py.*)", result.stdout)
    return "\n".join(test_cases)

@timeout(10)
def run_pytest(
    input_code: str,
    pytest_code: str,
    tmp_folder: str = "tmp",
    random_subdir: bool = False,
    clean_test: bool = True,
):
    current_dir = os.getcwd()
    tmp_dir_path = os.path.join(current_dir, tmp_folder)

    pytest_code = re.sub("\s+\\n", "\n", pytest_code)
    print(pytest_code)

    if random_subdir:
        random_name = "".join(
            random.choices(string.ascii_letters + string.digits, k=20)
        )
        tmp_dir_path = os.path.join(tmp_dir_path, random_name)
    os.makedirs(tmp_dir_path, exist_ok=True)

    solution_file_path = os.path.join(tmp_dir_path, "source.py")
    with open(solution_file_path, "w") as solution_file:
        solution_file.write(input_code)

    test_file_path = os.path.join(tmp_dir_path, "test_source.py")
    with open(test_file_path, "w") as test_file:
        test_file.write(pytest_code)

    try:
        result = subprocess.run(
            [
                "pytest",
                "--cov=source",
                test_file_path,
                "--cov-report",
                "term-missing",
                "-vv",
            ],
            capture_output=True,
            text=True,
            cwd=tmp_dir_path,
            timeout=5,
        )
    except Exception as e:
        print("Failed to run tests:", str(e))
        return {"coverage": 0, "failed_assertions": False, "stderr": e, "stdout": ""}

    result.stdout = strip_ansi_escape_sequences(result.stdout)

    # COVERAGE
    coverage_percent = get_coverage_percent(result)

    # NUMBER OF FAILED TESTS
    num_fails = get_num_fails(result)

    # FAIL DETAILS
    fails = get_test_case_fails(result)

    # TMP FILE CLEANUP
    if clean_test:
        shutil.rmtree(tmp_dir_path, ignore_errors=True)

    return {
        "coverage": coverage_percent,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "failed_assertions": num_fails,
        "fails": fails,
    }
