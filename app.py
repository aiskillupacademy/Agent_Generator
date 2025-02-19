from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
import os
import streamlit as st

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

llm_google = get_llm()

class Inputs(BaseModel):
    inputs: dict[str, str] = Field(description="inputs as keys and their description as values.")

class Workflow(BaseModel):
    workflow: dict[str, dict[str, str]] = Field(description="steps as keys and their inputs and description as sub dictionary keys and values.")

def get_system_prompt(desc):
    """
    Generates a system prompt instructing a specialist to complete a specific task based on a query.

    Parameters:
        desc (dict): A dictionary containing the query as input.
        llm (LLMChain): The language model chain to process the prompt.

    Returns:
        str: The generated system prompt.
    """
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert in creating clear, concise, and purpose-driven prompts for professionals.  
        Write a system prompt that instructs a specialist to complete a specific task based on a given query.  
        The query will be provided as input.  

        The system prompt should clearly specify the specialist's role and provide detailed instructions to complete the task effectively, aligning with the context and requirements of the query.  
        Include all necessary details to ensure the task is completed accurately, efficiently, and in line with the query's intent.  

        Query: {query}  

        You can start with:  
        'You are an Expert [Content type or domain-specific] specialist...', 'You are a [Content type or domain-specific] professional with 20 years of experience...', 'You are a tech-savvy enthusiast skilled in [Content type/task domain]...'  
        Don't give a header or footer. Never start with 'Task Completion Prompt...'. No emojis. No bold.  

        Identify the domain or context specific to the query and specify the role clearly at the beginning of the prompt.  
        Avoid generic roles; instead, use specific ones such as 'Healthcare Operations Specialist,' 'Technology Product Strategist,' 'Finance Data Analyst,' etc. 
        """
    )
    chain = prompt | llm_google
    system_prompt = chain.invoke(desc).content
    return system_prompt

def get_human_prompt(desc):
    """
    Generates a human prompt that provides a clear and specific task to be completed.

    Parameters:
        desc (dict): A dictionary containing the query as input.
        llm (LLMChain): The language model chain to process the prompt.

    Returns:
        str: The generated human prompt.
    """
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert in creating clear, concise, and actionable instructions for humans.  
        Write a human prompt that provides a clear and specific task to be completed based on a given query.  
        The query will be provided as input.  

        The human prompt should be clear, task-focused, and easy to understand.  
        Include necessary details, but avoid overloading with information.  

        Query: {query}  

        Example start: 'Please write...', 'Your task is to...', 'Provide a...'  

        Avoid abstract or overly general instructions. Make the task actionable, with clear deliverables or objectives. Never start with 'Here is your prompt...'. Just give the prompt.
        """
    )
    chain = prompt | llm_google
    human_prompt = chain.invoke(desc).content
    return human_prompt

def get_inputs(agent_name, agent_details):
    prompt = ChatPromptTemplate.from_template(
    """
    You are given a tool name and its details. Your task is to determine the essential inputs required from the user to operate the tool effectively.  

    ### **Guidelines:**  
    - Identify only the necessary inputs for the tool to function correctly.  
    - Keep the response concise, structured, and relevant.  
    - Provide a clear input name and a brief description of its purpose.  
    - Don't write about budget, landing URL.

    ---

    ### **Tool Name:** {name}  
    ### **Tool Details:** {detail}  

    #### **Required Inputs:**  
    (Generate a structured comma separated string of inputs with their names and descriptions.)  

    Keep it short and to the point.

    \n{inst}
    """)

    parser = PydanticOutputParser(pydantic_object=Inputs)
    chain = prompt | llm_google | parser
    result = chain.invoke({"name": agent_name, "detail": agent_details, "inst": parser.get_format_instructions()}).inputs

    return result

def get_workflow(agent_name, agent_details, result):
    prompt1 = ChatPromptTemplate.from_template(
    """
    You are given a tool name, its details, and the required inputs needed to operate it.  
    Your task is to generate a workflow (all the steps) for the tool. 

    ### **Guidelines:**  
    - Ensure the steps follow a logical and actionable sequence.  
    - Clearly outline each step with concise phrasing.   
    - Keep the response structured, professional, and easy to follow.
    - Do not include steps for collecting inputs — focus only on the workflow itself. 
    - Define the inputs for all the steps from the required inputs list provided. There can be more than 1 input for a step.

    ---

    ### **Tool Name:** {name}  
    ### **Tool Details:** {detail}    
    ### **Required Inputs:** {inputs} 

    #### **Steps to Complete the Task:**  
    (Generate a clear, structured step-by-step guide.)  
    Keep it short and to the point.
    - Start the steps after all necessary inputs have already been provided.

    \n{inst}
    """)

    parser1 = PydanticOutputParser(pydantic_object=Workflow)
    chain1 = prompt1 | llm_google | parser1
    result1 = chain1.invoke({"name": agent_name, "detail": agent_details, "inputs": result, "inst": parser1.get_format_instructions()}).workflow

    return result1

# Streamlit app
st.title("Agent Generator")

# Input query
agent_name = st.text_input("Enter agent name")
agent_details = st.text_input("Enter agent description")

if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = {}

if st.button("Generate"):
    chain1 = ChatPromptTemplate.from_template(f"Given the agent name and detail, create a 2 line description about the agent on what it does. \n Agent Name: {agent_name} \n\n Agent Detail: {agent_details}") | llm_google
    st.session_state.description = chain1.invoke({}).content
    st.session_state.inputs = get_inputs(agent_name, agent_details)
    st.session_state.workflow = get_workflow(agent_name, agent_details, st.session_state.inputs)

    st.session_state.prompts = []
    for keys in st.session_state.workflow.keys():
        prmpts = {
            'step': keys,
            'description': st.session_state.workflow[keys]['description'],
            'inputs': st.session_state.workflow[keys]['inputs'].split(','),
            "system_prompt": get_system_prompt(f"Task: {keys}. \n Desc: {st.session_state.workflow[keys]['description']}"),
            "human_prompt": get_human_prompt(f"Task: {keys}. \n Desc: {st.session_state.workflow[keys]['description']}")
        }
        st.session_state.prompts.append(prmpts)

# Display inputs and workflow if generated
if "inputs" in st.session_state and "description" in st.session_state:
    st.header(agent_name.upper())
    st.write(st.session_state.description)
    st.subheader("INPUTS")
    for key, value in st.session_state.inputs.items():
        st.markdown(f"**{key}:** {value}")

    st.header("WORKFLOW")
    for p in st.session_state.prompts:
        st.markdown(f"### {p['step']}")
        st.write(f"**Description:** {p['description']}")
        st.write(f"**Inputs Required:** {', '.join(p['inputs'])}")
        with st.expander("System Prompt"):
            st.write(p['system_prompt'])
        with st.expander("Human Prompt"):
            st.write(p['human_prompt'])

# User input fields (persist using session_state)
if st.button("Try it out..."):
    st.session_state.show_inputs = True

if st.session_state.get("show_inputs", False):
    for key, value in st.session_state.inputs.items():
        st.session_state.user_inputs[key] = st.text_input(label=key, help=value, value=st.session_state.user_inputs.get(key, ""))

# Run Agent Logic
if st.button('Run Agent'):
    final_input = ""
    for p in st.session_state.prompts:
        st.write(p['step'])
        inputs_text = "\n\n".join([f"{i}: {st.session_state.user_inputs.get(i, '')}" for i in p['inputs']])
        
        s_p = p['system_prompt']
        h_p = p['human_prompt'] + "\n\n" + inputs_text + "\n\n" + final_input

        chain = ChatPromptTemplate.from_messages([("system", s_p), ("user", h_p)]) | llm_google
        res = chain.invoke({}).content
        st.write(f"Output: {res}")

        final_input = res  # Store result for next iteration