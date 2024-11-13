from langchain.embeddings import CacheBackedEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import CSVLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from yaml import safe_load

if __name__ == "__main__":
    # load resume data into Documents
    loader = CSVLoader(file_path="UpdatedResumeDataSet.csv")
    resume_documents = loader.load()

    # resume_documents = resume_documents[:100]
    for doc in resume_documents:
        doc.metadata["search_method"] = "faiss"

    # split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    resume_chunks = text_splitter.split_documents(resume_documents)

    # create local vectorstore
    underlying_embeddings = OpenAIEmbeddings()
    store = LocalFileStore("./cache/")

    # Create cached embeddings
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, store, namespace=underlying_embeddings.model
    )

    # Create the vector store
    db = FAISS.from_documents(documents=resume_chunks, embedding=cached_embedder)

    # create retrievers
    k_results = 100

    # similarity search
    faiss_retriever = db.as_retriever(search_kwargs={"k": k_results})

    # keyword
    for doc in resume_documents:
        doc.metadata["search_method"] = "bm25"
    bm25_retriever = BM25Retriever.from_documents(resume_documents)
    bm25_retriever.k = k_results

    composite_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5],
        id_key="row"
    )

    with open("job_spec.yml", 'r') as file:
        data = safe_load(file)

    results_must_have = composite_retriever.invoke(data["must have"])
    results_role = composite_retriever.invoke(data["role match"])
    results_about = composite_retriever.invoke(data["about you"])
    results_tech_stack = composite_retriever.invoke(data["tech stack"])

    must_have_set = set([result.metadata.get('row', []) for result in results_role])
    role_set = set([result.metadata.get('row', []) for result in results_role])
    about_set = set([result.metadata.get('row', []) for result in results_about])
    teck_stack_set = set([result.metadata.get('row', []) for result in results_tech_stack])
    matches_all_set = role_set & about_set & teck_stack_set

    print(f"must have: {must_have_set}")
    print(f"matches role: {role_set}")
    print(f"matches about: {about_set}")
    print(f"matches tech stack: {teck_stack_set}")
    print(f"matches all: {matches_all_set}")

    llm = ChatOpenAI(model_name='gpt-4o', temperature=0)
    SCREEN_PROMPT = PromptTemplate.from_template("""
    You are helping me screen resumes. Here is what a candidate must have:
    {criteria}
    
    Here is the candidate's resume/cv:
    {resume}
    Does the candidate meet this criteria? (yes/no/perhaps) Please explain your answer in one sentence.
    """)
    screening_chain = SCREEN_PROMPT | llm | StrOutputParser()

    # screen the first 5 for must have criteria
    for result in results_must_have[:20]:
        row = result.metadata.get("row", 0)
        resume = resume_documents[row].page_content
        criteria = data["must have"]
        print(f"{row}: {screening_chain.invoke(input={'criteria': criteria, 'resume': resume})}")
