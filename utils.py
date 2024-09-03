import re
import pynvml

import pandas as pd
import altair as alt
import streamlit as st

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

# @st.fragment(run_every=1)
def render_gpu_monitor():
    pass
    # utilization_text = st.empty()
    # progress_bar = st.progress(0)
    # total, used, free = get_gpu_memory_info()
    # utilization = used / total
    # utilization_text.text(f"Memory Utilization: {utilization:.2%}")
    # progress_bar.progress(int(utilization * 100))


def visualize_pytest_results(pytest_results):
    st.subheader("Pytest Results")
    cov_col, num_fails_col, distrib_col = st.columns([0.33, 0.33, 0.33])
    all_passes = re.findall("test_source.py::.*PASSED", pytest_results["stdout"])
    all_errors = re.findall("\\nFAILED test_source.py::[^\\n]*", pytest_results["stdout"])

    with cov_col:
        st.metric("Test Coverage", f"{pytest_results['coverage']}%")

    with num_fails_col:
        st.metric("Failed Assertions", pytest_results['failed_assertions'])

    with distrib_col:
        chart_data = pd.DataFrame({
            'Category': ['Passed', 'Failed'],
            'Count': [len(all_passes), len(all_errors)]
        })

        chart = alt.Chart(chart_data).mark_bar().encode(
            x='Category',
            y='Count',
            color=alt.condition(
                alt.datum.Category == 'Passed',
                alt.value('green'),
                alt.value('red')
            )
        ).properties(
            width=300,
            height=200,
            title='Test Results'
        )

        st.altair_chart(chart)

    if pytest_results['stdout']:
        results_col, msg_col = st.columns([0.5, 0.5])

        with results_col:
            st.subheader("Test Overview")
            for passed_test in all_passes:
                st.success(passed_test)

            for error in all_errors:
                st.error(error)

        with msg_col:
            st.subheader("Standard Output")
            st.text(pytest_results['stdout'])

        st.text(pytest_results["stderr"])