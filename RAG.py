from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langsmith import Client

import os
import tempfile
import requests
from typing import Literal
from dotenv import load_dotenv

class RAG:
    def __init__(self):
        load_dotenv()

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        client = Client() # LangSmith tracing

        self.directory = tempfile.TemporaryDirectory()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, streaming=True)
        self.contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        self.rephraser = RunnableBranch(
            (
                lambda x: not x.get("chat_history", False),
                self.set_qry,
            ),
            self.contextualize_prompt | self.llm | StrOutputParser()
        )

        self.set_system_prompt("You are an assistant for question-answering tasks. "
                                "Use the following pieces of retrieved context to answer the question. "
                                "If you don't know the answer, just say that you don't know."
                                "\n\n{context}")

        class GradeDocuments(BaseModel):
            binary_score: Literal["yes", "no"] = Field(
                description="Documents are relevant to the question, 'yes' or 'no'"
            )
        self.grader_llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0).with_structured_output(GradeDocuments)
        self.filter_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a grader assessing relevance of a "
                "retrieved document to a user question. If the document "
                "is at all related in any way, grade it as relevant. "
                "It should not be a stringent test. The goal is "
                "simply to filter out erroneous retrievals. Give a "
                "binary score 'yes' or 'no' score to indicate relevance."),
                ("human", "User question: {question} \n\n Retrieved document: \n\n {document}")
            ]
        )
        self.doc_grader = self.filter_prompt | self.grader_llm

        self.cnt = 0

        self.store = {}

        self.user_query = None

        self.filter_flag = True
        self.truncate_chat_hist = True
        self.history_length = 10

        self.urls = []
    
    def set_system_prompt(self, prompt):
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        self.qa_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)
    
    def get_session_history(self, session_id):
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        if self.truncate_chat_hist and len(self.store[session_id].messages) >= self.history_length:
            if len(self.store[session_id].messages) == 7:
                del self.store[session_id].messages[0]
            del self.store[session_id].messages[:2]
        return self.store[session_id]

    def filter_documents(self, docs):
        if not self.filter_flag:
            return docs
        filtered = []
        for doc in docs:
            result = self.doc_grader.invoke({"question": self.user_query, "document": doc.page_content})
            if result.binary_score == "yes":
                filtered.append(doc)
        return filtered
    
    def set_qry(self, qry):
        if type(qry) != str:
            qry = qry["input"]
        self.user_query = qry
        return qry
        
    def update_vector_store(self):
        self.loader = DirectoryLoader(self.directory.name, loader_cls=UnstructuredMarkdownLoader)
        self.docs = self.loader.load()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        self.all_splits = self.text_splitter.split_documents(self.docs) 
        self.vectorstore = Chroma.from_documents(documents=self.all_splits, embedding=OpenAIEmbeddings())
        self.retriever = self.vectorstore.as_retriever()
        
        self.history_aware_retriever = self.rephraser | RunnableLambda(self.set_qry) | self.retriever | self.filter_documents
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.qa_chain)
        self.conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        ).with_config({
            "configurable": {"session_id": self.directory.name}
        })
        
    def scrape_url(self, url):
        response = requests.get("https://r.jina.ai/" + url)

        if response.status_code == 200:
            self.cnt += 1

            with open(os.path.join(self.directory.name, str(self.cnt) + ".md"), 'w', encoding = "utf-8") as f:
                f.write(response.text)
        else:
            print(f"Failed to retrieve data. HTTP Status code: {response.status_code}")
    
    def web_search(self, qry, num_results):
        self.urls = []
        url = "https://google-search74.p.rapidapi.com/"
        querystring = {"query": qry, "limit": num_results,"related_keywords": "false"}
        headers = {
            "x-rapidapi-key": os.getenv("X_RAPIDAPI_KEY"),
            "x-rapidapi-host": "google-search74.p.rapidapi.com"
        }
        response = requests.get(url, headers=headers, params=querystring)
        for result in response.json()['results']:
            self.urls.append(result['url'])
            self.scrape_url(result['url'])
        
        self.update_vector_store()
        
    def query(self, qry):
        for (i, chunk) in enumerate(self.conversational_rag_chain.stream({"input": qry})):
            if i >= 2:
                print(chunk['answer'], end="", flush=True)
        print("")

    def cleanup(self):
        self.directory.cleanup()
