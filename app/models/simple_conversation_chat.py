import threading

# from langchain import ConversationChain
from langchain.callbacks.manager import CallbackManager
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from app.callbacks.streaming import ThreadedGenerator, ChainStreamHandler

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.llms import LlamaCpp 
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class SimpleConversationChat:
    def __init__(self, history):
        self.memory = ConversationBufferMemory(return_messages=True)
        self.set_memory(history)

    def set_memory(self, history):
        for message in history:
            if message.role == 'assistant':
                self.memory.chat_memory.add_ai_message(message.content)
            else:
                self.memory.chat_memory.add_user_message(message.content)

    def generator(self, user_message):
        g = ThreadedGenerator()
        threading.Thread(target=self.llm_thread, args=(g, user_message)).start()
        return g

    def llm_thread(self, g, user_message):
        try:
            streamingHandler = ChainStreamHandler(g)
            llm = ChatOpenAI(
                verbose=True,
                streaming=True,
                callback_manager=CallbackManager([ChainStreamHandler(g)]),
                temperature=0.7,
            )
            '''
            llm = LlamaCpp(
                #model_path="./models/vicuna-13b-v1.5-16k.Q4_K_M.gguf",
                model_path="./models/vicuna-7b-v1.5.Q4_K_M.gguf",
                #model_path="./models/llama-2-7b-chat.q4_K_M.gguf",
                #model_path="./models/ELYZA-japanese-Llama-2-7b-fast-instruct-q4_K_M.gguf",
                verbose=True,
                n_ctx=4096,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                streaming=True, 
            )'''

            embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
            vectorstore = Chroma(persist_directory="./chroma_issues_512", embedding_function=embeddings)
            retriver = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)
            #print(retriver)

            docs = retriver.get_relevant_documents(user_message)
            docs = docs + retriver.get_relevant_documents(user_message)
 
            response = "\n"
            response = response + "| ID | Subject | Content |\n"
            response = response + "|    -: |    :-: |   :-  |\n"
            for doc in docs:
                content = doc.page_content
                subject = doc.metadata['subject']
                id = int(doc.metadata['id'])
                url = f"https://gate.tok.access-company.com/redmine/issues/{id}"
                response = response + f"| [{id}]({url})|    {subject}  |   {content}   |\n"
            response = response + "\n"
            
            # 検索結果を回答する
            streamingHandler.on_llm_new_token(response)

        finally:
            g.close()
