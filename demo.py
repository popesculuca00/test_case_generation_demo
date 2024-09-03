import streamlit as st
st.set_page_config(layout="wide",
                   page_title="Test case generation demo"
                   )


from generation_page_components import render_title, render_selector_container, render_code_windows
from model_wrapper import  bg_init_all_models
from constants import code_snippets, EMPTY_TEST_MSG
from utils import visualize_pytest_results



def init_session_state():
    if "model" not in st.session_state.keys():
        st.session_state["model"] = None #ModelInference()

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


def main():
    render_title()
    with st.container():
        selected_snippet, selected_model, cache_erase_col = render_selector_container()
    for _ in range(2):
        st.text("")
    with st.container():
        clear_col = render_code_windows(cache_erase_col, selected_snippet, selected_model)
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