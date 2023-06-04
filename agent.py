# This script sets up the agent employed by the bot, For demostration, the agent
# in this script acts as a shop assistant at Home Depot.

# Set up the environment

import os
from dotenv import find_dotenv, load_dotenv
import flatdict  # flattens nested dict
import re

# packages for web scraping/ searching
import validators
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time  # gives extra time for webdriver to load webpage
from serpapi import GoogleSearch

# Language models and embedding models
# from langchain.llms import OpenAI  # my OpenAI trial expired :/
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.llms import Cohere
from langchain.llms import AI21
from langchain.embeddings import CohereEmbeddings

# Chains and chain components
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory

# Load api key variables from .env file and set api keys
load_dotenv(find_dotenv('private/.env'))
SERPAPI_API_KEY = os.environ["SERPAPI_API_KEY"]
AI21_API_KEY = os.environ["AI21_API_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

# Set language model
# llm = OpenAI(temperature=0)
llm = AI21(temperature=0.5)
llm_doc = AI21(temperature=0.1)  # for queries related to docs
# Rate limit for Cohere model is 5 calls/min
# llm = Cohere(temperature=0.5)
# llm_doc = Cohere(temperature=0)  # for queries related to docs


################################################################################


def genius(query,db):
    """
    The agent.
    """

    # Prompt guidelines

    guidelines = """

    If the query is not related to services or products offered by Home Depot, say you can't help.
    If the query is asking about a specific product available at Home Depot, look for the product.
    If the query is not about a specific product, offer the customer some general advice and suggest a few related Home Depot products.
    If you don't know the answer, say you don't know and provide contact information for customer service.

    """

    # Set prompt template

    prompt_template = """
    You are a helpful shop assistant at Home Depot. This is a query from a Home Depot customer: {query}

    Answer the query following these guidelines: {guidelines}

    In your answer:
    Explain your answer briefly.
    When you mention specific products, use full product names.
    If you are asked about one specific product and you found the product, provide a link to that product.

    """

    prompt = PromptTemplate(template=prompt_template, 
                            input_variables=["query", "guidelines"])


    ############################################################################


    # Chains and functions

    prompt_str = prompt.format(query=query, guidelines=guidelines)

    chain = RetrievalQA.from_chain_type(llm=llm_doc, chain_type="stuff",
                                        retriever=db.as_retriever())

    # Cheeck url
    def check_url(url):
        valid=validators.url(url)
        if valid==True:
            try:
                headers = {"User-Agent": 
                           "Mozilla/5.0 (X11; CrOS x86_64 12871.102.0) \
                           AppleWebKit/537.36 (KHTML, like Gecko) \
                           Chrome/81.0.4044.141 Safari/537.36"}       
                response = requests.head(url,headers=headers)         
                if response.status_code == 200:
                    return True
                else:
                    return False
            except requests.ConnectionError as e:
                return False
        else:
            return False

    # Create tools from functions

    @tool
    def is_homedepot(query):
        """
        Decide whether the query is asking about a product available at Home Depot 
        query : customer query
        """
        prompt_new = 'Is the following asking about a product available at Home Depot? ' + query
        chain_new = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                                retriever=db.as_retriever())
        ans = chain_new.run(prompt_new)
        return ans

    @tool
    def get_keyword(query):
        """
        Get the search keyword from the query.
        query : customer query
        """
        prompt_new = 'What is the product mentioned in the query? ' + query
        chain_new = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", 
                                                retriever=db.as_retriever())
        ans = chain_new.run(prompt_new) 
        return ans

    @tool
    def get_products(keyword):
        """
        Searches for the keyword at Home Depot. Add results to vector store.
        keyword : Search keyword
        """

        params = {
        "engine": "home_depot",
        "api_key": SERPAPI_API_KEY,
        "q": keyword
        }
        search = GoogleSearch(params)

        if "products" in search.get_dict().keys():
            products = search.get_dict()["products"]
        else:
            ans = chain.run(prompt_str)
            return ans
        
        ########################################################################    
        # Clean the search results

        # drop unused product info
        for p in products:
            for key in ['position', 'thumbnails', 'serpapi_link', 'collection', 'variants']:
                if key in p.keys():
                    del p[key]
        
        # Embedding models only work on texts or numbers.
        # Fix values other than str, int or float.
        for p in products:
            for k,v in p.items():
                # Change value to 'none' if it's not str/int/float 
                #   or list/bool (will take care of these later)
                if type(v) not in [str, int, float, list, bool]:
                    p[k] = 'none'
                # Convert list of strings to string
                if type(v)==list:
                    p[k] = '-'.join(v)
                # Convert boolean to string
                if type(v)==bool:
                    if p[k]==True:
                        p[k] = 'true'
                    else:
                        p[k] = 'false'

        # Vector stores do not accept nested dict as metadata. Need to flatten it.
        for p in range(len(products)):
            # flatten each dict in products
            p_flatdict = flatdict.FlatDict(products[p], delimiter='.')
            # convert flatdict back to dict
            p_dict = {}
            for i in p_flatdict.iteritems():
                p_dict.update({i})
                products[p] = p_dict
        
        ########################################################################
        # Add search results to vector store

        # Get a list of product names (will be used as the texts for vector stores)
        product_names = [p['title'] for p in products]

        # Add texts to vector store
        nonlocal db  # assess db outside the function
        db.add_texts(product_names, metadatas=products)

        ans = chain.run(prompt_str)
        return ans

    @tool
    def get_details(link):
        """
        Get product details from the link. Add results to vector store.
        link : link to product detail page
        """

        # Validate link url
        if not check_url(link): return "Please provide more information."
        
        # Use webdriver to load the page
        driver = webdriver.Chrome('./chromedriver') 
        driver.get(link)    
        time.sleep(20)  # give extra time to ensure that the page is fully rendered    
        page = driver.page_source

        # Scrape texts on page
        soup = BeautifulSoup(page, "html.parser")
        name = soup.find('div', {'class': 'product-details__badge-title--wrapper'}).text
        ids = soup.find('div', {'class': 'sui-flex sui-text-xs sui-flex-wrap'}).text
        overview = soup.find('section', {'id': 'product-section-product-overview'}).text
        specs = soup.find('section', {'id': 'specifications-desktop'}).text

        # Close webdriver after scraping
        driver.close()

        # clean the data
        overview = re.sub(r"(\w)([A-Z])", r"\1 \2", overview)
        specs = re.sub(r"(\w)([A-Z])", r"\1 \2", specs)
        specs = re.sub('See Similar Items', ' ', specs)
        ids = re.sub('Internet', ' Internet:', ids)
        ids = re.sub('Model', ' Model:', ids)
        ids = re.sub('Store SKU', ' Store SKU:', ids)
        ids = re.sub('Store SO SKU', ' Store SO SKU:', ids)

        # Concatenate scraped texts into a string
        pdp = 'Product Name: ' + name + '\nProduct IDs: ' + ids 
              + '\nProduct Overview: ' + overview + '\nSpecifications: ' + specs

        # Add text to vector store
        nonlocal db  # assess db outside the function
        db.add_texts(pdp)
        
        ans = chain.run(prompt_str)
        return ans


    ############################################################################


    # Create an agent with tools

    tools = [

        Tool(
            name = "is_homedepot",
            func=is_homedepot,
            description="Use this tool if you are not sure the query is asking \
                        about a specific product available at Home Depot. \
                        Input is the query",        
        ),

        Tool(
            name = "get_products",
            func=get_products,
            description="Look for the products on Home Depot's website. \
                        Input is a string.",
            return_direct=True
        ),

        Tool(
            name = "get_details",
            func=get_details,
            description="Find the product details about a specific product. \
                The input is the link to the product. You can get this link \
                from the search results you got from the function 'get_products'",
            return_direct=True
        )

    ]

    memory = ConversationBufferMemory(memory_key="chat_history")
    agent = initialize_agent(tools, llm, 
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                             verbose=False, memory=memory)
    return agent.run(query)
