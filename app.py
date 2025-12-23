import streamlit as st
from transformers import pipeline

st.title("Python Code Error Explainer (Free LLM)")

# Input fields
code = st.text_area("Paste your Python code here")
error_message = st.text_input("Paste the error message here")

# Button to analyze
if st.button("Explain Error"):
    if not code or not error_message:
        st.warning("Please enter both code and error message.")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Load free model (small instruct model)
                generator = pipeline(
                    "text-generation", 
                    model="mosaicml/mpt-7b-instruct", 
                    device=0  # 0 for GPU, -1 for CPU
                )

                prompt = f"""
                You are an expert Python tutor.
                Explain this error in simple words and suggest how to fix it.

                Code:
                {code}

                Error:
                {error_message}
                """

                # Generate explanation
                result = generator(prompt, max_length=200)[0]['generated_text']
                st.success("Explanation:")
                st.write(result)

            except Exception as e:
                st.error(f"Error occurred: {e}")
