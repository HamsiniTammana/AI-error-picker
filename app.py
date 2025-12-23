import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

st.title("Python Code Error Explainer (Free LLM, Works for All Errors)")

# Inputs
code = st.text_area("Paste your Python code here")
error_message = st.text_input("Paste the error message here")

if st.button("Explain Error"):
    if not code or not error_message:
        st.warning("Please enter both code and error message.")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Load small model (CPU-friendly)
                tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
                model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
                generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

                prompt = f"""
                You are a Python tutor. Explain this error in simple words and suggest how to fix it.

                Code:
                {code}

                Error:
                {error_message}
                """

                result = generator(prompt, max_length=200, do_sample=False)[0]['generated_text']
                st.success("Explanation:")
                st.write(result)

            except Exception as e:
                st.error(f"Error occurred: {e}")
