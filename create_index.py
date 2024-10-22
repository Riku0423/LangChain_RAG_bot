# 必要なモジュールをインポートします
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter
import os
from dotenv import load_dotenv

# .envファイルから環境変数を読み込みます
load_dotenv()

# 現在のファイルのディレクトリパスを取得します
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# テキストファイルをロードするためのTextLoaderを初期化します
# エンコーディングを明示的に指定します
loader = TextLoader(os.path.join(BASE_DIR, "data", "Dify Docs.txt"), encoding="utf-8")

# テキストファイルの内容をロードし、分割します
documents = loader.load_and_split()

# トークンベースのテキスト分割器を初期化します
text_splitter = TokenTextSplitter(
    chunk_size=200,
    chunk_overlap=10,
    encoding_name="cl100k_base",
    add_start_index=True,
)

# ドキュメントを分割します
splitted_docs = text_splitter.transform_documents(documents)

# OpenAIの埋め込みモデルを初期化します
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# 分割されたドキュメントからFAISSベクトルストアを作成します
vector_store = FAISS.from_documents(splitted_docs, embedding_model)

# ディレクトリを作成してDocument, Embedding情報を格納します
output_dir = os.path.join(BASE_DIR, "vector_store")
os.makedirs(output_dir, exist_ok=True)
vector_store.save_local(os.path.join(output_dir, "vector_store.json"))

print("インデックスとチャンクが作成され、保存されました。")
