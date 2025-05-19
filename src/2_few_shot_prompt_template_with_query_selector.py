from pathlib import Path
from dotenv import load_dotenv
import os

# from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate,FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

env_path = Path('.').parent / '.env'
print(f"Loading environment variables from {env_path}")
if not env_path.exists():
    raise FileNotFoundError(f"Environment file {env_path} not found. Please create a .env file with the required variables.")
else:
    load_dotenv(dotenv_path=env_path)

AZURE_OPEN_API_KEY = os.getenv("AZURE_OPEN_API_KEY")
AZURE_OPEN_API_ENDPOINT = os.getenv("AZURE_OPEN_API_ENDPOINT")
AZURE_OPEN_API_VERSION = os.getenv("AZURE_OPEN_API_VERSION")
AZURE_EMBEDDING_API_ENDPOINT = os.getenv("AZURE_EMBEDDING_API_ENDPOINT")
AZURE_EMBEDDING_VERSION = os.getenv("AZURE_EMBEDDING_VERSION")


if not AZURE_OPEN_API_KEY or not AZURE_OPEN_API_ENDPOINT or not AZURE_OPEN_API_VERSION:
    raise ValueError("Please set the AZURE_OPEN_API_KEY, AZURE_OPEN_API_ENDPOINT, and AZURE_OPEN_API_VERSION environment variables.")
else:
    print("Environment variables loaded successfully.")


# model_name='google/flan-t5-base'


# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

hf_embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

#Initialize prompt template
example_prompt = PromptTemplate.from_template(
    # input_variables=["article", "news_type"],
    "article: {article}\nnews_type: {news_type}")

# Set the examples
examples =  [
    {
        "question": "Who lived longer, Muhammad Ali or Alan Turing?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali
""",
    },
    {
        "question": "When was the founder of craigslist born?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952
""",
    },
    {
        "question": "Who was the maternal grandfather of George Washington?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball
""",
    },
    {
        "question": "Are both the directors of Jaws and Casino Royale from the same country?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate Answer: The director of Jaws is Steven Spielberg.
Follow up: Where is Steven Spielberg from?
Intermediate Answer: The United States.
Follow up: Who is the director of Casino Royale?
Intermediate Answer: The director of Casino Royale is Martin Campbell.
Follow up: Where is Martin Campbell from?
Intermediate Answer: New Zealand.
So the final answer is: No
""",
    },
]

examples=[
    {"article":"Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again.",
     "news_type":"2"},
    {"article":"Oil prices soar to all-time record, posing new menace to US economy (AFP) AFP - Tearaway world oil prices, toppling records and straining wallets, present a new economic menace barely three months before the US presidential elections.",
     "news_type":"2"},
    {"article":'Dreaming done, NBA stars awaken to harsh Olympic reality (AFP) AFP - National Basketball Association players trying to win a fourth consecutive Olympic gold medal for the United States have gotten the wake-up call that the "Dream Team" days are done even if supporters have not.', 
     "news_type":"1"},
    {"article":"Tiger Runs Out of Steam After Storming Start  KOHLER, Wisconsin (Reuters) - Tiger Woods failed to make  the most of a red-hot start in the U.S. PGA Championship third  round on Saturday, having to settle for a three-under-par 69.", 
     "news_type":"1"},
    {"article":"Hacker Cracks Apple's Streaming Technology (AP) AP - The Norwegian hacker famed for developing DVD encryption-cracking software has apparently struck again  #151; this time breaking the locks on Apple Computer Inc.'s wireless music streaming technology.", 
     "news_type":"3"},
     {"article":"Building Dedicated to Columbia Astronauts (AP) AP - A former dormitory converted to classrooms at the Pensacola Naval Air Station was dedicated Friday to two Columbia astronauts who were among the seven who died in the shuttle disaster Feb. 1, 2003.", 
     "news_type":"3"}
]

# print(example_prompt.invoke(examples[0]).to_string())

# Initialize the fewshot prompt template
# prompt = FewShotPromptTemplate(
#     examples = examples,
#     example_prompt = example_prompt,
#     prefix = "You are a helpful assistant. Answer the question based on the information provided.\n",
#     suffix = "\n\nQuestion: {input}\nAnswer:",
#     input_variables = ["input"],
# )

# print(
#     prompt.invoke({"input": "Who was the father of Mary Ball Washington?"}).to_string()
# )

# Initialize the semantic search query selector
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=hf_embedding,
    vectorstore_cls=Chroma,
    k=1,
    example_keys=["article", "news_type"],       #filter only these fields
)

###################################################################
# embeddings = AzureOpenAIEmbeddings(
#     api_key=AZURE_OPEN_API_KEY,  
#     api_version=AZURE_EMBEDDING_VERSION,
#     azure_endpoint=AZURE_EMBEDDING_API_ENDPOINT,
#     deployment="text-embedding-ada-002",
#     model="text-embedding-ada-002",
#     )

# example_selector = SemanticSimilarityExampleSelector.from_examples(
#     examples=examples,
#     embeddings=embeddings,
#     vectorstore_cls=Chroma,
#     example_prompt=example_prompt,
#     k=1
# )
# input_text = "The meaning of life is 42"
# vector = embeddings.embed_query(text=input_text)
# print(vector[:3])


########################################################




# Select the most similar example to the input.
# question = "Google IPO faces Playboy slip-up The bidding gets underway for Google's public offering, despite last-minute worries over an interview with its bosses in Playboy magazine."
# selected_examples = example_selector.select_examples({"article": question})
# print(f"Examples most similar to the input: {question}")

# for example in selected_examples:
#     print("\n")
#     for k, v in example.items():
#         print(f"{k}: {v}")


prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="article: {input}",
    input_variables=["article"],
)

print(
    prompt.invoke({"article": "Dollar Falls Broadly on Record Trade Gap  NEW YORK (Reuters) - The dollar tumbled broadly on Friday  after data showing a record U.S. trade deficit in June cast  fresh doubts on the economy's recovery and its ability to draw  foreign capital to fund the growing gap."}).to_string()
)




