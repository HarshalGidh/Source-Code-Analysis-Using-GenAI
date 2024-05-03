#Importing Libraries
from langchain.vectorstores import Chroma
from src.helper import load_embedding
from dotenv import load_dotenv
import os
from src.helper import repo_ingestion
from flask import Flask, render_template, jsonify, request
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI
import google.generativeai as genai
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


app = Flask(__name__)


embeddings = load_embedding()
persist_directory = "db"
# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)

# Creatring Retriever object
retriever = vectordb.as_retriever() 

# llm = ChatOpenAI()

# Configure generativeai with your API key
llm = ChatGoogleGenerativeAI(model="gemini-pro",
                 temperature=0.7, top_p=0.85,google_api_key=GOOGLE_API_KEY)

memory = ConversationSummaryMemory(llm=llm, memory_key = "chat_history", return_messages=True)

# qa = ConversationChain(llm=llm,memory=memory,verbose=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k":3}), memory=memory)

llm_prompt_template = """You are an assistant for question-answering tasks.
Use the following context to answer the question.
If you don't know the answer, just say that you don't know.
Use five sentences maximum and keep the answer concise.\n
Question: {question} \nContext: {context} \nAnswer:"""

llm_prompt = PromptTemplate.from_template(llm_prompt_template)


# Combine data from documents to readable string format.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)





@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html') # to run the UI



@app.route('/chatbot', methods=["GET", "POST"])
def gitRepo():

    if request.method == 'POST':
        user_input = request.form['question']
        repo_ingestion(user_input) # Data Ingestion 
        os.system("python store_index.py") #to run store_index to create db 

    return jsonify({"response": str(user_input) })


# to chat with the user :
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)

    if input == "clear":
        os.system("rm -rf repo")


    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | llm_prompt
    | llm
    | StrOutputParser()
    )

    return rag_chain.invoke(input)
    # result = qa(input)
    # print(result['answer'])
    # return str(result["answer"])



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)