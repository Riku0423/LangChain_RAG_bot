import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamlitCallbackHandler
from langchain.schema import Document
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

st.title("LangChain Retrieval Bot")

# 埋め込みモデルを初期化
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# FAISSインデックスを読み込む
index_path = os.path.join("vector_store", "vector_store.json")
vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# チャットモデルを初期化
llm = ChatOpenAI(
    model=os.getenv("OPENAI_API_MODEL"),
    temperature=float(os.getenv("OPENAI_API_TEMPERATURE", 0)),
    streaming=True
)

# メモリを初期化
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# クエリ生成用のプロンプトテンプレート
query_generator_template = """ユーザーの質問に基づいて、関連情報を検索するための3つの異なるクエリを生成してください。
これらのクエリは、元の質問の意図を捉えつつ、異なる角度から情報を探索するものであるべきです。名詞のみの端的なクエリを生成してください。

ユーザーの質問: {question}

生成されたクエリ:
1.
2.
3.
"""
query_generator_prompt = PromptTemplate(
    input_variables=["question"],
    template=query_generator_template
)

# クエリ生成用のLLMChain
query_generator_chain = LLMChain(llm=llm, prompt=query_generator_prompt)

# 回答生成用のPromptTemplate
answer_template = """
質問に対して、以下のコンテキストを使用して回答してください。

質問: {question}

コンテキスト:
{context}

回答:
"""
answer_prompt = PromptTemplate(template=answer_template, input_variables=["question", "context"])

# 回答生成用のLLMChain
answer_chain = LLMChain(llm=llm, prompt=answer_prompt)

# チャットインターフェースを設定
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("質問を入力してください")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # クエリを生成
        generated_queries = query_generator_chain.run(prompt)
        queries = generated_queries.strip().split("\n")
        
        # 検索を実行
        combined_docs = []
        for query in queries:
            docs = vectorstore.similarity_search(query, k=3)
            combined_docs.extend(docs)
        
        # コンテキストを作成
        context = "\n\n".join([f"ソース: {doc.metadata['source']}\n内容: {doc.page_content}" for doc in combined_docs])
        
        # 回答を生成
        callback = StreamlitCallbackHandler(st.container())
        response = answer_chain.run(question=prompt, context=context, callbacks=[callback])
        
        # 生成されたクエリと検索結果を表示するエクスパンダー
        with st.expander("生成されたクエリと検索結果"):
            for i, query in enumerate(queries, 1):
                st.subheader(f"クエリ {i}: {query}")
            for i, doc in enumerate(combined_docs, 1):
                st.markdown(f"**ドキュメント {i}:**")
                st.markdown(f"**ソース:** {doc.metadata['source']}")
                st.markdown(f"**内容:** {doc.page_content}")
                st.markdown("---")

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
