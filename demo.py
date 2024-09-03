
import sys
from threading import Thread
import re
import streamlit as st
st.set_page_config(layout="wide")

import altair as alt
import pandas as pd
from pytest_runner import run_pytest

from model_wrapper import ModelInference
from constants import code_snippets, available_models, EMPTY_TEST_MSG
from constants import extract_code, get_gpu_memory_info


def init_session_state():
    if "model" not in st.session_state.keys():
        st.session_state["model"] = None#ModelInference()

    if "start_generation" not in st.session_state.keys():
        st.session_state["start_generation"] = False

    if "generated_pytest" not in st.session_state.keys():
        st.session_state["generated_pytest"] = EMPTY_TEST_MSG

    if "pytest_results" not in st.session_state.keys():
        st.session_state["pytest_results"] = None

    if "cnt_code" not in st.session_state.keys():
        st.session_state["cnt_code"] = list(code_snippets.keys())[0]



@st.cache_resource
def get_model(model_name):
    return ModelInference(model_name)


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


def main():
    _, title_col, _ = st.columns([0.4, 0.3, 0.3])
    with title_col:
        st.title("Unit test generator")


    source_col, test_col = st.columns(2)

    with source_col:
        st.subheader("source.py")
        code_selector_col, cache_erase_col = st.columns([0.82, 0.18])

        with code_selector_col:
            selected_snippet = st.selectbox(
                "Select a code snippet or choose 'Custom':",
                list(code_snippets.keys()),
                label_visibility="collapsed",
                placeholder="Select a code snippet or choose 'Custom'",
                index=None
            )

        if st.session_state["model"]:
            with cache_erase_col:
                if st.button("Empty Cache"):
                    st.session_state["model"].empty_cache()
                    st.rerun()

        initial_code = code_snippets[selected_snippet]

        if selected_snippet == "Custom":
            user_code = st.text_area(
                        "input code",
                        label_visibility="collapsed",
                        value="",
                        height=300,
                        key="code_input"
                    )
        else:
            st.code(
                initial_code,
            )
            user_code = code_snippets[selected_snippet]

        
    with test_col:
        st.subheader("test_source.py")
        model_selector_col, gpu_monitor_col = st.columns([0.6, 0.4])

        with model_selector_col:
            selected_model = st.selectbox(
                "Select a model:",
                list(available_models.keys()),
                label_visibility="collapsed",
                placeholder="Select a model",
                index=None
            )

        with gpu_monitor_col:
            utilization_text = st.empty()
            progress_bar = st.progress(0)
            total, used, free = get_gpu_memory_info()
            utilization = used / total
            utilization_text.text(f"Memory Utilization: {utilization:.2%}")
            progress_bar.progress(int(utilization * 100))

        with st.spinner("Loading model.."):
            if  available_models[selected_model]:
                if (not st.session_state["model"]) or available_models[selected_model] != st.session_state["model"].model_path:
                    del st.session_state["model"]
                    st.session_state["model"] = get_model(available_models[selected_model])
                    st.session_state["pytest_results"] = None
                    st.session_state["generated_pytest"] = EMPTY_TEST_MSG
                    st.rerun()

        with st.spinner("Generating code.."):
            if st.session_state["start_generation"]:
                st.session_state["generated_pytest"] = ""

                streaming_placeholder = st.empty()

                for token in st.session_state["model"].generate_stream(st.session_state["start_generation"]):
                    streaming_placeholder.code(st.session_state["generated_pytest"] + token)
                    st.session_state["generated_pytest"] += token
                print("Done generating!")

                
                st.session_state["generated_pytest"] = extract_code(st.session_state["generated_pytest"])
                st.session_state["model"].last_response = st.session_state["generated_pytest"] 
                streaming_placeholder.empty()
                st.code(st.session_state["generated_pytest"], language="python", line_numbers=True)

                st.session_state["start_generation"] = False
            else:
                st.code(st.session_state["generated_pytest"], language="python", line_numbers=True)


    generate_col, run_col, clear_col, _ = st.columns([0.1, 0.1, 0.1, 0.7])

    if st.session_state["model"]:
        with generate_col:
            if st.button("Generate test"):
                st.session_state["generated_pytest"] != "# You need to provide an input code"
                if user_code:
                    st.session_state["start_generation"] = user_code
                else:
                    st.warning("Please enter some code in the input area.")
                st.rerun()

    
    if st.session_state["generated_pytest"] != EMPTY_TEST_MSG:
        with run_col:
            if st.button("Run Pytest"):
                st.session_state["pytest_results"] = run_pytest(user_code, st.session_state["generated_pytest"])
                st.rerun()

    if st.session_state["pytest_results"]:
        visualize_pytest_results(st.session_state["pytest_results"])

        with clear_col:
            if st.button("Clear Results"):
                st.session_state["pytest_results"] = None
                st.rerun()

    if selected_snippet != st.session_state["cnt_code"]:
        print("Resetting test results")
        st.session_state["cnt_code"] = selected_snippet
        st.session_state["pytest_results"] = None
        st.session_state["generated_pytest"] = EMPTY_TEST_MSG
        st.rerun()

if __name__ == "__main__":
    init_session_state()
    main()