import streamlit as st
from transformers import pipeline

# Page Config
st.set_page_config(
    page_title="VeriFact-IN",
    page_icon="🛡️",
    layout="centered"
)

# Load Model
@st.cache_resource
def load_model():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

classifier = load_model()

# Header
st.markdown("<h1 style='text-align: center;'>🛡️ VeriFact-IN</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>AI-Based Fake News & AI-Generated Content Detection</h4>",
    unsafe_allow_html=True
)

st.write("---")

# Input Box
text = st.text_area(
    "📰 Enter News Text (Hindi / English)",
    height=180,
    placeholder="यहाँ न्यूज़ टेक्स्ट या English news paste करें..."
)

# Button
if st.button("🔍 Analyze News"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        labels = [
            "fake news",
            "real news",
            "ai generated",
            "human written"
        ]

        with st.spinner("Analyzing with AI model..."):
            result = classifier(text, labels)

        st.success("✅ Analysis Complete")

        # Results Display
        st.subheader("📊 Prediction Results")
        for label, score in zip(result["labels"], result["scores"]):
            st.progress(score)  #(int(score * 100))
            st.write(f"**{label.upper()}** : {round(score * 100, 2)} %")

# Footer
st.write("---")

st.markdown(
    "<p style='text-align: center;'>Prototype Model | VeriFact-IN | AI Project</p>",
    unsafe_allow_html=True
)