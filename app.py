import streamlit as st
import fitz
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="TenderMind AI",
    page_icon="📄",
    layout="wide"
)


# --------------------------------
# CUSTOM DESIGN
# --------------------------------

st.markdown("""
<style>

/* Main website background */
.stApp {
    background-color: #f4f7fc !important;
    color: #1e293b !important;
}

/* Main content text */
[data-testid="stAppViewContainer"] .main {
    color: #1e293b !important;
}

/* Headings */
h1, h2, h3, h4 {
    color: #142c52 !important;
}

/* Normal paragraphs */
p {
    color: #334155 !important;
}

/* Labels */
label {
    color: #1e293b !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #14213d !important;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* Buttons */
.stButton button {
    background-color: #1d4ed8;
    color: white !important;
    border-radius: 10px;
    font-weight: bold;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #142c52 !important;
}

[data-testid="stMetricLabel"] {
    color: #334155 !important;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# HEADER
# --------------------------------

st.markdown(
    '<div class="main-title">📄 TenderMind AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Tender Analysis & Response Preparation System</div>',
    unsafe_allow_html=True
)

st.write("")

st.info(
    "Upload a tender, extract requirements, "
    "find supporting evidence and prepare responses."
)


# --------------------------------
# PDF TEXT EXTRACTION
# --------------------------------

def extract_pdf(uploaded_file):

    document = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text()

        if text.strip():

            pages.append({
                "text": text,
                "page": page_number + 1
            })

    return pages


# --------------------------------
# REQUIREMENT EXTRACTION
# --------------------------------

def extract_requirements(pages):

    requirements = []

    pattern = re.compile(
        r'\b(must|shall|required|should)\b',
        re.IGNORECASE
    )

    for page in pages:

        sentences = re.split(
            r'(?<=[.!?])\s+',
            page["text"]
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if len(sentence) > 20 and pattern.search(sentence):

                requirements.append({

                    "Requirement ID":
                    f"R{len(requirements)+1:03}",

                    "Requirement":
                    sentence,

                    "Category":
                    get_category(sentence),

                    "Page":
                    page["page"]

                })

    return requirements


# --------------------------------
# REQUIREMENT CATEGORY
# --------------------------------

def get_category(sentence):

    text = sentence.lower()

    if any(word in text for word in
           ["security", "access control", "encryption"]):

        return "Security"

    elif any(word in text for word in
             ["support", "maintenance", "helpdesk"]):

        return "Support"

    elif any(word in text for word in
             ["certification", "compliance", "iso"]):

        return "Compliance"

    elif any(word in text for word in
             ["integration", "cloud", "platform", "system"]):

        return "Technical"

    elif any(word in text for word in
             ["experience", "eligibility", "turnover"]):

        return "Eligibility"

    else:

        return "General"


# --------------------------------
# EVIDENCE EXTRACTION
# --------------------------------

def extract_evidence(documents):

    evidence = []

    for document_name, pages in documents:

        for page in pages:

            sentences = re.split(
                r'(?<=[.!?])\s+',
                page["text"]
            )

            for sentence in sentences:

                sentence = sentence.strip()

                if len(sentence) > 20:

                    evidence.append({

                        "text": sentence,

                        "document": document_name,

                        "page": page["page"]

                    })

    return evidence


# --------------------------------
# AI EVIDENCE MATCHING
# --------------------------------

def match_evidence(requirements, evidence):

    results = []

    if not evidence:

        for req in requirements:

            results.append({

                **req,

                "Status": "Evidence not found",

                "Evidence": "No supporting document found.",

                "Source": "None",

                "Draft Response":
                "Please verify with the relevant department."

            })

        return results

    documents = [
        item["text"] for item in evidence
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    all_text = [
        req["Requirement"] for req in requirements
    ] + documents

    matrix = vectorizer.fit_transform(all_text)

    requirement_matrix = matrix[:len(requirements)]

    evidence_matrix = matrix[len(requirements):]

    similarities = cosine_similarity(
        requirement_matrix,
        evidence_matrix
    )

    for index, req in enumerate(requirements):

        best_index = similarities[index].argmax()

        score = similarities[index][best_index]

        best_evidence = evidence[best_index]

        if score >= 0.18:

            status = "Evidence found - Verify"

            evidence_text = best_evidence["text"]

            source = (
                f'{best_evidence["document"]}, '
                f'Page {best_evidence["page"]}'
            )

            draft = (
                "The supporting company document states: "
                f'"{evidence_text}"'
            )

        elif score >= 0.07:

            status = "Potential evidence - Review"

            evidence_text = best_evidence["text"]

            source = (
                f'{best_evidence["document"]}, '
                f'Page {best_evidence["page"]}'
            )

            draft = (
                "Potential supporting evidence found. "
                "Employee verification is required."
            )

        else:

            status = "Evidence not found"

            evidence_text = "No supporting evidence found."

            source = "None"

            draft = (
                "No compliance claim generated. "
                "Ask the relevant department to verify."
            )

        results.append({

            **req,

            "Status": status,

            "Evidence": evidence_text,

            "Source": source,

            "Draft Response": draft

        })

    return results


# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.title("TenderMind AI")

st.sidebar.markdown("### Upload Tender")

tender_file = st.sidebar.file_uploader(
    "Upload Tender PDF",
    type=["pdf"]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Company Evidence Library")

evidence_files = st.sidebar.file_uploader(
    "Upload Supporting Documents",
    type=["pdf"],
    accept_multiple_files=True
)


# --------------------------------
# DASHBOARD
# --------------------------------

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 📑 Tender Analysis")

    st.write(
        "Extract individual requirements "
        "from uploaded tender documents."
    )

with col2:

    st.markdown("### 🔍 Evidence Intelligence")

    st.write(
        "Find supporting information "
        "from approved company documents."
    )


# --------------------------------
# ANALYSE BUTTON
# --------------------------------

if st.button(
    "🚀 Analyse Tender",
    use_container_width=True
):

    if tender_file is None:

        st.error("Please upload a tender PDF.")

    else:

        with st.spinner("Analysing tender document..."):

            tender_pages = extract_pdf(tender_file)

            requirements = extract_requirements(
                tender_pages
            )

            company_documents = []

            for file in evidence_files or []:

                pages = extract_pdf(file)

                company_documents.append(
                    (file.name, pages)
                )

            evidence = extract_evidence(
                company_documents
            )

            results = match_evidence(
                requirements,
                evidence
            )

            st.session_state["results"] = results

            st.success(
                "Tender analysis completed successfully!"
            )


# --------------------------------
# DISPLAY RESULTS
# --------------------------------

if "results" in st.session_state:

    results = st.session_state["results"]

    total = len(results)

    found = sum(
        r["Status"].startswith("Evidence found")
        for r in results
    )

    potential = sum(
        r["Status"].startswith("Potential")
        for r in results
    )

    missing = sum(
        r["Status"] == "Evidence not found"
        for r in results
    )

    st.markdown("---")

    st.markdown("## 📊 Tender Analysis Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Requirements", total)

    c2.metric("Evidence Found", found)

    c3.metric("Needs Review", potential)

    c4.metric("Missing Evidence", missing)

    st.markdown("---")

    st.markdown("## 📋 Extracted Requirements")

    if not results:

        st.warning(
            "No requirements detected. "
            "Check whether the PDF contains selectable text."
        )

    else:

        dataframe = pd.DataFrame(results)

        st.dataframe(
            dataframe[
                [
                    "Requirement ID",
                    "Requirement",
                    "Category",
                    "Page",
                    "Status"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        st.markdown("## 🔎 Evidence Verification")

        for result in results:

            with st.expander(
                f'{result["Requirement ID"]} - '
                f'{result["Category"]}'
            ):

                st.write("**Requirement:**")

                st.write(result["Requirement"])

                st.write("**Status:**")

                st.write(result["Status"])

                st.write("**Supporting Evidence:**")

                st.write(result["Evidence"])

                st.write("**Source:**")

                st.write(result["Source"])

                st.write("**Draft Response:**")

                st.write(result["Draft Response"])

        st.markdown("---")

        st.markdown("## 📥 Export Draft Response")

        draft = "# TenderMind AI - Draft Response\n\n"

        draft += (
            "For employee review only. "
            "Not a compliance guarantee.\n\n"
        )

        for result in results:

            draft += (
                f'## {result["Requirement ID"]}\n\n'
            )

            draft += (
                f'**Requirement:** '
                f'{result["Requirement"]}\n\n'
            )

            draft += (
                f'**Status:** '
                f'{result["Status"]}\n\n'
            )

            draft += (
                f'**Draft Response:** '
                f'{result["Draft Response"]}\n\n'
            )

            draft += (
                f'**Evidence:** '
                f'{result["Evidence"]}\n\n'
            )

            draft += (
                f'**Source:** '
                f'{result["Source"]}\n\n'
            )

            draft += "---\n\n"

        st.download_button(

            label="⬇️ Download Draft Response",

            data=draft,

            file_name="TenderMind_Response.md",

            mime="text/markdown",

            use_container_width=True

        )


# --------------------------------
# FOOTER
# --------------------------------

st.markdown("---")

st.caption(
    "TenderMind AI | AI-Assisted Tender Management | "
    "Human verification required before submission."
)