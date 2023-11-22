# (위의 코드3줄은 해당 이슈로 인해 적어줘야 함 : https://discuss.streamlit.io/t/issues-with-chroma-and-sqlite/47950)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



# 필요한 module import 
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button

# buy me a coffee 배너 띄우기 (for 수익화)
button(username="kerin07", floating=True, width=221)

#제목
st.title("Communication with PDF")
st.write("---")

#OpenAI KEY 유저에게 입력 받기 (비용X)
openai_key = st.text_input('OPEN_AI_API_KEY', type="password")

st.write("---") # st.write() 에서 마크다운 문법사용가능, '---': 구분선 기능 
st.subheader('pdf를 넣으면 pdf에 대해 질문할 수 있어요 :sunglasses: ', divider='rainbow')

#파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 업로드해주세요!",type=['pdf']) #type 옵션은 업로드 파일 pdf로 제한두겠다는 것 
st.write("---")

# 업로드하는 pdf 파일을 임시로 둘 공간이 필요
def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

#업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    #Split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    #Embedding
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)

    # load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    #Question
    st.header("PDF에게 질문해보세요!!")
    question = st.text_input('질문을 입력하세요')

    if st.button('질문하기'):
        with st.spinner('로딩중...'):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_key)
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
            result = qa_chain({"query": question})
            st.write(result["result"])
