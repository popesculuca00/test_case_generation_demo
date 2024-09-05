import streamlit as st

from constants import code_snippets, available_models, EMPTY_TEST_MSG
from model_wrapper import get_model
from pytest_runner import run_pytest
from utils import extract_code, visualize_pytest_results


def render_title():
    _, title_col, _ = st.columns([0.4, 0.3, 0.3])
    with title_col:
        st.title("Unit test generator")
    for _ in range(4):
        st.text("")


def render_selector_container():
    source_col, test_col = st.columns(2)
    with source_col:

        _, subheader_col, _ = st.columns([0.2, 0.3, 0.5])
        with subheader_col:
            st.subheader("source.py")

        selected_snippet = st.selectbox(
            "Select a code snippet or choose 'Custom':",
            list(code_snippets.keys()),
            label_visibility="collapsed",
            placeholder="Select a code snippet or choose 'Custom'",
            index=None
        )

    with test_col:
        _, subheader_col, _ = st.columns([0.2, 0.4, 0.4])
        with subheader_col:
            st.subheader("test_source.py")

        if st.session_state["loaded_models"] is None:
            model_selector_col, model_loading_col = st.columns([0.75, 0.25])
        else: 
            model_selector_col, model_loading_col = st.columns([0.83, 0.17])

        with model_selector_col:
            selected_model = st.selectbox(
                "Select a model:",
                list(available_models.keys()),
                label_visibility="collapsed",
                placeholder="Select a model",
                index=None,
                disabled= not st.session_state["loaded_models"]
            )   
    return selected_snippet, selected_model, model_loading_col


def render_generation_controls(user_code):
    generate_col, run_col, clear_col = st.columns([0.3, 0.3, 0.3])

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
    return clear_col


def render_code_windows(selected_snippet, selected_model):
    input_code_col, generation_col = st.columns([0.5, 0.5])

    st.markdown("""
    <style>
    div.stSpinner > div {
        text-align:right;
        align-items: right;
        justify-content: right;
    }
    </style>""", unsafe_allow_html=True)

    with st.spinner("Fetching code.."):
        with input_code_col:
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
                    initial_code, language="python", line_numbers=True
                )
                user_code = code_snippets[selected_snippet]
            clear_col = render_generation_controls(user_code)

    with generation_col:
        if  available_models[selected_model]:
            if (not st.session_state["model"]) or available_models[selected_model] != st.session_state["model"].model_path:
                del st.session_state["model"]
                st.session_state["model"] = get_model(available_models[selected_model])
                st.session_state["pytest_results"] = None
                st.session_state["generated_pytest"] = EMPTY_TEST_MSG
                st.rerun()

        if st.session_state["start_generation"]:
            st.session_state["generated_pytest"] = ""
            streaming_placeholder = st.empty()

            with st.spinner("Generating code.."):
                for token in st.session_state["model"].generate_stream(st.session_state["start_generation"]):    
                    streaming_placeholder.code(st.session_state["generated_pytest"] + token, language="python", line_numbers=True)
                    st.session_state["generated_pytest"] += token

            extracted_code = extract_code(st.session_state["generated_pytest"])      
            if extracted_code != st.session_state["generated_pytest"]:
                st.session_state["generated_pytest"] = extracted_code
                st.session_state["model"].last_response = st.session_state["generated_pytest"]
                streaming_placeholder.empty()
                st.code(st.session_state["generated_pytest"], language="python", line_numbers=True)
            st.session_state["start_generation"] = False
            st.rerun()
        else:
            st.code(st.session_state["generated_pytest"], language="python", line_numbers=True)

    if st.session_state["pytest_results"]:
        visualize_pytest_results(st.session_state["pytest_results"])
        with clear_col:
            if st.button("Clear Results"):
                st.session_state["pytest_results"] = None
                st.rerun()



