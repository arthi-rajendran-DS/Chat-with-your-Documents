import streamlit as st
import faiss
from faiss import swigfaiss
from haystack.nodes import FARMReader, PreProcessor, PDFToTextConverter, DensePassageRetriever
from haystack.nodes import BM25Retriever
from haystack.document_stores import FAISSDocumentStore
from haystack.utils import launch_es
from haystack.pipelines import ExtractiveQAPipeline

document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")
document_store.save(index_path="my_index.faiss", config_path="my_config.json")

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="word",
    split_length=100,
    split_respect_sentence_boundary=True,
    split_overlap=3
)

@st.cache(show_spinner=False, allow_output_mutation=True)
def written_document(pdf_file):
    converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=["en"])
    document = [converter.convert(file_path=pdf_file.name, meta=None)[0]]
    preprocessed_docs = preprocessor.process(document)
    document_store.write_documents(preprocessed_docs)
    return None

@st.cache(show_spinner=False, allow_output_mutation=True)
def predict(question, pdf_file):
    written_document(pdf_file)
    retriever = BM25Retriever(document_store)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
    pipe = ExtractiveQAPipeline(reader, retriever)
    result = pipe.run(query=question, params={"Retriever": {"top_k": 20}, "Reader": {"top_k": 5}})
    answers = print_answers(result)
    return answers

def print_answers(results):
    fields = ["answer", "score"]
    answers = results["answers"]
    filtered_answers = []
    
    for ans in answers:
        filtered_ans = {
            field: getattr(ans, field)
            for field in fields
            if getattr(ans, field) is not None
        }
        filtered_answers.append(filtered_ans)

    return filtered_answers

st.set_page_config(page_title="Ponniyin Selvan Chatbot", page_icon=":guardsman:", layout="wide")

st.title("Ponniyin Selvan Chatbot")
st.markdown("<center><b>Sample Questions:</b> Who is Vandia Devan?</center>", unsafe_allow_html=True)

question = st.text_area("Ask an open question!", height=100)

pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

if st.button("Ask"):
    if not pdf_file:
        st.warning("Please upload a PDF file.")
    elif not question:
        st.warning("Please enter a question.")
    else:
        answers = predict(question, pdf_file)
        if answers:
            st.markdown("<h2>Answers:</h2>", unsafe_allow_html=True)
            for ans in answers:
                st.write(f"Answer: {ans['answer']}")
                st.write(f"Score: {ans['score']}")
                st.write("---")
        else:
            st.warning("No answer found.")
