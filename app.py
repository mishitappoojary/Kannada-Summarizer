import streamlit as st
import pyperclip
from extractive_summary import extractive_summary
from abstractive_summary import summarize_text
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Transliterate Kannada text to Romanized text
def romanize_text(text):
    return transliterate(text, sanscript.KANNADA, sanscript.ITRANS)

def main():
    st.title("Kannada Text Summarizer")
    text = st.text_area("Enter Kannada text:", height=200)
    
    # Display letter count of the input text
    st.write(f"Input text letter count: {len(text)}")
    
    # Option to choose summarization type
    summary_type = st.radio("Select Summary Type:", ["Extractive", "Abstractive"])
    
    # Initialize session state for summary_text and summary_type if not already done
    if 'summary_text' not in st.session_state:
        st.session_state.summary_text = ''
    if 'summary_type' not in st.session_state:
        st.session_state.summary_type = ''

    # Extractive summarization only: Show the option to select number of lines
    if summary_type == "Extractive":
        # Initialize session state for numberOfLines
        if 'numberOfLines' not in st.session_state:
            st.session_state.numberOfLines = 2  # Default value

        # List of options for number of lines
        lines_options = [2, 3, 4, 5, 10, 15]

        # Ensure the stored numberOfLines is in the range of options
        if st.session_state.numberOfLines not in lines_options:
            st.session_state.numberOfLines = 2  # Reset to default if out of range

        # Select the number of lines to summarize to
        st.session_state.numberOfLines = st.selectbox(
            "Summarize to how many lines: ",
            lines_options,
            index=lines_options.index(st.session_state.numberOfLines)  # Use index based on current value
        )

    if st.button("Summarize"):
        if text:
            # Count the number of sentences in the input text
            num_input_lines = len([sentence for sentence in text.split('.') if sentence.strip()])

            if summary_type == "Extractive":
                if st.session_state.numberOfLines > num_input_lines:
                    st.error("The number of lines requested for summarization exceeds the number of lines in the input text.")
                else:
                    with st.spinner("Generating extractive summary, please wait..."):
                        try:
                            extractive_summary_result = extractive_summary(text, num_sentences=st.session_state.numberOfLines)
                            st.session_state.summary_text = extractive_summary_result
                            st.session_state.summary_type = "Extractive"
                        
                        except Exception as e:
                            st.error(f"An error occurred while generating the extractive summary: {e}")
            
            elif summary_type == "Abstractive":
                with st.spinner("Generating abstractive summary, please wait..."):
                    try:
                        abstractive_summary_result = summarize_text(text)
                        st.session_state.summary_text = abstractive_summary_result
                        st.session_state.summary_type = "Abstractive"
                    
                    except Exception as e:
                        st.error(f"An error occurred while generating the abstractive summary: {e}")
        else:
            st.error("Please enter some text for summarization.")

    # Show the summary if it exists in session state
    if st.session_state.summary_text:
        st.subheader(f"{st.session_state.summary_type} Summary:")
        st.write(st.session_state.summary_text)

        # Romanized Text
        romanized_text = romanize_text(st.session_state.summary_text)
        st.markdown(f"*{romanized_text}*")

        # Display letter count of the summary
        st.write(f"Summary letter count: {len(st.session_state.summary_text)}")

        # Copy to clipboard functionality using pyperclip
        if st.button('Copy Summary to Clipboard'):
            pyperclip.copy(st.session_state.summary_text)
            st.success("Summary copied to clipboard!")

if __name__ == "__main__":
    main()
