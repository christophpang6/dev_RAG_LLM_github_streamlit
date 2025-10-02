import os
import streamlit as st
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st

HF_API_KEY = st.secrets["HF_API_KEY"]

# ======== Sample Training Documents ========
enhanced_sample_texts = {
    "space_missions.txt": """
    The Apollo 11 mission was the first successful attempt to land humans on the Moon. 
    It was launched on July 16, 1969, from Kennedy Space Center in Florida, atop a Saturn V rocket. 
    The spacecraft consisted of three main parts: the Command Module "Columbia," piloted by Michael Collins; 
    the Lunar Module "Eagle," which carried Neil Armstrong and Buzz Aldrin to the Moon’s surface; 
    and the Service Module, which supported the Command Module. 

    On July 20, 1969, at 20:17 UTC, the Lunar Module Eagle landed in the Sea of Tranquility. 
    Neil Armstrong became the first human to set foot on the Moon, uttering the famous words 
    "That’s one small step for man, one giant leap for mankind," followed by Buzz Aldrin soon after. 
    Michael Collins remained in lunar orbit aboard Columbia, maintaining communications and preparing for return. 

    The astronauts collected 47.5 pounds (21.5 kilograms) of lunar rock and soil samples, 
    conducted scientific experiments including seismometers and solar wind detectors, 
    and placed an American flag and a plaque that read: "We came in peace for all mankind." 
    The mission lasted 8 days, 3 hours, 18 minutes, and 35 seconds, concluding with splashdown 
    in the Pacific Ocean on July 24, 1969. The success of Apollo 11 fulfilled President John F. Kennedy’s 1961 
    goal of "landing a man on the Moon and returning him safely to the Earth" before the decade’s end. 
    """,

    "landmarks_architecture.txt": """
    The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France, 
    along the Seine River. It was designed by engineer Gustave Eiffel and constructed between 1887 and 1889 
    as the entrance arch to the 1889 Exposition Universelle (World’s Fair), which celebrated the centennial 
    of the French Revolution. At 324 meters (1,063 feet) tall, it was the tallest man-made structure 
    in the world until the completion of the Chrysler Building in New York City in 1930. 

    Originally criticized by artists and intellectuals in France for its radical design, 
    the Eiffel Tower eventually became a global cultural icon and one of the most visited paid monuments 
    in the world. The tower consists of three accessible levels: restaurants and shops on the first and second levels, 
    and an observation deck on the third level that offers panoramic views of Paris. 

    Constructed from over 18,000 iron parts joined by 2.5 million rivets, 
    the Eiffel Tower weighs approximately 10,100 tons. It requires regular repainting (every 7 years) 
    to prevent corrosion. It is illuminated nightly by 20,000 light bulbs, 
    making it one of the most recognizable symbols of France and modern engineering. 
    """,

    "programming_technologies.txt": """
    Python is a high-level, interpreted programming language created by Guido van Rossum in 1989 
    and first released in 1991. Its design emphasizes readability, simplicity, and efficiency, 
    famously using significant whitespace (indentation) rather than braces or keywords 
    to define code blocks. Python supports multiple programming paradigms including 
    object-oriented, functional, and procedural programming. 

    The language has an extensive standard library, often described as having 
    "batteries included," which provides modules for tasks such as file I/O, networking, 
    mathematics, and web services. Python is widely used in data science, machine learning, 
    artificial intelligence, scientific computing, web development, and automation. 
    Its ecosystem includes major libraries like NumPy, pandas, TensorFlow, PyTorch, Django, and Flask. 

    Python is currently maintained by the Python Software Foundation. 
    There are two main versions: Python 2 (now deprecated) and Python 3, 
    which introduced improvements such as Unicode support and better syntax consistency. 
    Python’s popularity has grown rapidly, making it one of the most widely taught 
    programming languages in universities and one of the top languages used in industry. 
    """,

    "science_discoveries.txt": """
    Penicillin, the world’s first widely used antibiotic, was discovered in 1928 by Scottish scientist Alexander Fleming. 
    While working at St. Mary’s Hospital in London, Fleming observed that a mold called Penicillium notatum 
    had contaminated one of his Petri dishes and killed the surrounding colonies of Staphylococcus bacteria. 
    This accidental discovery revealed that the mold produced a substance with powerful antibacterial properties. 

    Although Fleming published his findings in 1929, it took over a decade before penicillin was developed 
    into a usable medical treatment. In the early 1940s, a team of scientists including Howard Florey, Ernst Boris Chain, 
    and Norman Heatley at Oxford University succeeded in mass-producing penicillin, 
    supported by large-scale industrial efforts in the United States during World War II. 

    Penicillin revolutionized medicine by providing a reliable treatment for bacterial infections such as pneumonia, 
    syphilis, gonorrhea, scarlet fever, and wound infections. It saved countless lives during WWII, 
    dramatically reducing deaths from infected wounds among soldiers. The discovery of penicillin 
    marked the beginning of the modern antibiotic era, although overuse in subsequent decades 
    has contributed to antibiotic resistance, one of today’s major public health challenges. 
    """,

    "historical_events.txt": """
    World War II was a global conflict lasting from September 1, 1939, to September 2, 1945, 
    involving more than 100 million people from over 30 countries. It began when Germany, 
    under Adolf Hitler, invaded Poland, prompting Britain and France to declare war. 
    The major Allied powers included the United States, the Soviet Union, the United Kingdom, and China, 
    while the main Axis powers were Germany, Italy, and Japan. 

    The war was fought across Europe, Africa, Asia, and the Pacific. Key events included the Battle of Britain (1940), 
    the German invasion of the Soviet Union (1941), the Japanese attack on Pearl Harbor (1941), 
    the Allied invasion of Normandy (D-Day, June 6, 1944), and the liberation of Nazi concentration camps. 
    The Holocaust, in which approximately six million Jews and millions of others including Poles, Romani, 
    disabled individuals, and political prisoners were murdered, remains one of history’s most tragic atrocities. 

    World War II ended in Europe with Germany’s unconditional surrender on May 8, 1945 (V-E Day). 
    The war concluded globally after the United States dropped atomic bombs on Hiroshima (August 6, 1945) 
    and Nagasaki (August 9, 1945), forcing Japan’s surrender on August 15, 1945, 
    which was formally signed on September 2, 1945, aboard the USS Missouri. 

    The aftermath of World War II reshaped the global order: 
    the United Nations was founded in 1945 to promote peace and security, 
    the Cold War emerged between the U.S. and the Soviet Union, 
    and many nations in Asia and Africa began movements toward decolonization. 
    The war caused an estimated 70–85 million deaths, making it the deadliest conflict in human history. 
    """
}


# ======== Prepare FAISS Index ========
embedder = SentenceTransformer("all-MiniLM-L6-v2")
corpus, sources = [], []
for src, text in enhanced_sample_texts.items():
    for line in text.strip().split("\n"):
        line = line.strip()
        if line:
            corpus.append(line)
            sources.append(src)

embeddings = embedder.encode(corpus, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# ======== System message enforcing detailed "I don't know" fallback ========
SYSTEM_MESSAGE = (
    "You are a helpful assistant. Only answer based on the provided context. "
    "If the context does not contain the answer, respond with: "
    "'I don't know is used to demonstrate that the chatbot will not hallucinate "
    "if it doesn't know based off of the retrieved context.'"
)

# ======== Streamlit UI Config ========
st.set_page_config(page_title="Chris RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Multi-Turn RAG Chatbot Demo")

st.markdown("Creator: **Christopher Pang**  🔗 [LinkedIn](https://www.linkedin.com/in/christopherpang)")

st.markdown(
    "This demo shows a Retrieval-Augmented Generation chatbot that pulls from X documents and uses embeddings + LLM to answer domain-specific questions.<br>"
    "Ask questions about these topics: **Apollo 11 space mission, Eiffel Tower, Python programming, World War II, Penicillin**.<br>"
    "If a question is asked that is not in the retrieval vector database, the chatbot will respond with: **\"I don't know.\"** instead of hallucinating/making something up",
    unsafe_allow_html=True
)

# making it very simple for someone to start to interact
st.markdown("### Try clicking on one of these:")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🚀 Apollo 11 crew"):
        st.session_state["trigger_chat"] = "Who were the crew members of Apollo 11?"

with col2:
    if st.button("🗼 Eiffel Tower year"):
        st.session_state["trigger_chat"] = "When was the Eiffel Tower built?"

with col3:
    if st.button("💻 Python origin"):
        st.session_state["trigger_chat"] = "Who created Python and when was it released?"


# Hugging Face token input
hf_token = HF_API_KEY

# Slider controls (like in Gradio)
# max_tokens = st.slider("Max new tokens", min_value=1, max_value=2048, value=512, step=1)
# temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
# top_p = st.slider("Top-p (nucleus sampling)", min_value=0.1, max_value=1.0, value=0.95, step=0.05)

# set these so it is simpler for someone to use
max_tokens = 512
temperature = 0.0
top_p = 0.95

# Initialize chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ======== RAG + HF Chat Function ========
def rag_respond(message, hf_token, history, system_message, max_tokens, temperature, top_p):
    client = InferenceClient(token=hf_token, model="openai/gpt-oss-20b")

    # Combine previous Q&A
    history_text = ""
    for q, a in history[-3:]:
        history_text += f"Previous Q: {q}\nPrevious A: {a}\n"

    # FAISS retrieval
    # known issue:
    # history[-3:] was used for multi turn follow up questions such as: where did apollo 11 go to?, then 
    # who went there? chat_history feed into retrieval_query is necessary to be able to have this
    # multi turn follow up questions functionality. 
    # however when switching topics, will have irrelevant retrieval chunks from previous topic relative
    # to the new topic and the LLM will then say I don't know.
    # can instead use history[-1:] to use less of the past chat_history while maintaining
    # history[-1:] still has topic switching causing i don't know output. however, it takes one
    # i don't know before topic is successfully switched while still enabling multi turn followup questions
    retrieval_query = " ".join([f"{q} {a}" for q, a in history[-1:]] + [message])
    q_emb = embedder.encode([retrieval_query], convert_to_numpy=True)
    D, I = index.search(q_emb, k=5)
    retrieved_chunks = [(corpus[i], sources[i], D[0][j]) for j, i in enumerate(I[0])]
    context_text = "\n".join([f"[{src}] {chunk}" for chunk, src, _ in retrieved_chunks])

    # Build messages for HF API
    messages_list = [{"role": "system", "content": system_message}]
    for q, a in history:
        messages_list.append({"role": "user", "content": q})
        messages_list.append({"role": "assistant", "content": a})
    messages_list.append({"role": "user", "content": f"{history_text}\nCurrent Query: {message}\nContext:\n{context_text}"})

    # Collect response
    response_text = ""
    for chunk in client.chat_completion(messages_list, max_tokens=max_tokens, stream=True, temperature=temperature, top_p=top_p):
        if len(chunk.choices) and chunk.choices[0].delta.content:
            response_text += chunk.choices[0].delta.content

    return response_text

# ======== Streamlit Chat Loop ========
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

# Always show chat input
user_input = st.chat_input("Ask me something...")

# Determine prompt: button click or typed input
prompt = st.session_state.pop("trigger_chat", None) or user_input

if prompt:
    if not hf_token:
        st.warning("Please enter your HuggingFace token above first.")
    else:
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_respond(prompt, hf_token, st.session_state.chat_history,
                                       SYSTEM_MESSAGE, max_tokens, temperature, top_p)
                st.write(response)
        st.session_state.chat_history.append((prompt, response))
