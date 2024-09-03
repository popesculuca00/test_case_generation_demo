import time
from threading import Thread
import streamlit as st
st.set_page_config(layout="wide")


from pytest_runner import run_pytest

from model_wrapper import ModelInference
from constants import code_snippets, available_models, EMPTY_TEST_MSG
from utils import extract_code, render_gpu_monitor, visualize_pytest_results


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

    if "loaded_models" not in st.session_state.keys():
        st.session_state["loaded_models"] = bg_init_all_models()

@st.cache_resource
def bg_init_all_models():
    all_models = {}
    for model_name, model_path in available_models.items():
        all_models[model_name] = ModelInference(model_path)
    return all_models

@st.cache_resource
def get_model(model_name):
    model_name = {j: i for (i, j) in available_models.items()}[model_name]

    while model_name not in st.session_state["loaded_models"].keys():
        time.sleep(3)
    return st.session_state["loaded_models"][model_name]

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
                    st.session_state["cnt_code"] = selected_snippet
                    st.session_state["pytest_results"] = None
                    st.session_state["generated_pytest"] = EMPTY_TEST_MSG
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
            render_gpu_monitor()

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
    print("AAAAAAAAAAAAAAAAA")
    init_session_state()
    main()