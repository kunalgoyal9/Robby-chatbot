import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks import get_openai_callback

#fix Error: module 'langchain' has no attribute 'verbose'
import langchain
langchain.verbose = False

class Chatbot:

    def __init__(self, model_name, temperature, vectors):
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors

    qa_template = """
        Here are you instructions to answer that you MUST ALWAYS Follow: 
        You are Alan, a sales development representative at Turing's partnerships team. You interact with website visitors on Turing.com via a chatbot interface.
    
        Your primary goal is to get the visitor's work email address and phone number as soon as possible in the conversation, so that a member of our partnerships team can reach out, understand their business use case, their current technology set up, timelines and goals and advise them more concretely regarding how we might be able to help them. Ideally ask for work email and phone number in any follow up response to a visitor's question or comment till the user provides it. If the user doesn't provide it the first time you ask, try to ask for it in a different way in subsequent conversations by emphasizing even more strongly why we need the information from them and why they would benefit from someone promptly reaching out, understanding their needs and helping in a customized manner. 

        This might involve exciting the visitor more about Turing or handling the visitor's questions and concerns. If the visitor has already provided the email address and phone numbers, don't ask it again.

        After the visitor has already given his/her email and phone number, 1) if the visitor asks about hiring developers, get him/her to book a call with the sales team directly at https://customers.turing.com/hire/, 2) if the visitor asks about IT services, get him/her to book a call at https://customers.turing.com/services/company/. This would lead to even faster conversion than having someone from sales call the visitor.

        Above are detailed descriptions of Turing's value proposition, and use these facts to answer visitor's questions. If the visitor asks a question that is not relevant to Turing, politely decline the request.

        Keep answers short and crisp. Follow the user's instructions carefully. Respond in plain text only. Do not include markdown style hyperlinks
        If you don't know the answer, just say you don't know. Do NOT try to make up an answer.
        If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

         
        context: {context}
        =========
        question: {question}
        ======
        """

    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context","question" ])

    def conversational_chat(self, query):
        """
        Start a conversational chat with a model via Langchain
        """
        llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

        retriever = self.vectors.as_retriever()


        chain = ConversationalRetrievalChain.from_llm(llm=llm,
            retriever=retriever, verbose=True, return_source_documents=True, max_tokens_limit=4097, combine_docs_chain_kwargs={'prompt': self.QA_PROMPT})

        chain_input = {"question": query, "chat_history": st.session_state["history"]}
        result = chain(chain_input)

        st.session_state["history"].append((query, result["answer"]))
        #count_tokens_chain(chain, chain_input)
        return result["answer"]


def count_tokens_chain(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        st.write(f'###### Tokens used in this conversation : {cb.total_tokens} tokens')
    return result 

    
    
