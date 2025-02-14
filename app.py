from typing import TypedDict, Dict, Annotated,Union, Callable
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.messages import BaseMessage
import operator
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain.prompts import PromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel
from typing import List
from langchain.tools import Tool, StructuredTool
from langchain_core.documents import Document
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, StateGraph
from langchain.output_parsers import PydanticOutputParser
import os
import streamlit as st

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

def get_llm():
    return ChatVertexAI(model="gemini-1.5-flash", temperature=0.3)

llm = get_llm()

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
    chain = prompt | get_llm()
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
    chain = prompt | get_llm()
    human_prompt = chain.invoke(desc).content
    return human_prompt

def quality_checker_bot(task, output):
    """
    Evaluates the quality of an output based on a given task and suggests improvements.

    Parameters:
        task (str): The task description.
        output (str): The generated output to be checked.
        llm (LLMChain): The language model chain to process the prompt.

    Returns:
        str: Suggested changes for improving the output or "No changes needed."
    """
    quality_check_prompt = PromptTemplate(
        input_variables=["task", "output"],
        template="""
        You are a quality-checking bot designed to improve outputs based on their alignment with the given task. 
        Analyze the provided output in detail and suggest only the changes needed to enhance its quality. 

        Task: {task}

        Output: {output}

        When making suggestions, consider:
        - Does the output fully address the task?
        - Is it accurate and relevant?
        - Is it clear, detailed, and well-structured?

        Provide your suggestions as a list of changes. 
        If no changes are needed, state: "No changes needed."

        Example Output:
        - Add more detail about [specific aspect].
        - Correct inaccuracies in [specific section].
        - Improve clarity by rephrasing [specific sentences].
        """
    )
    quality_chain = quality_check_prompt  | get_llm()
    result = quality_chain.invoke({"task": task, "output": output}).content
    return result

def generate_content(human_prompt, system_prompt="You are an expert in generating anything. Your task will be to follow the user instruction and generate accordingly.", temp=0.2, model="gemini-1.5-flash", platform="google"):
    """
    Generates text using a Large Language Model (LLM) with configurable parameters.

    Parameters:
        human_prompt (str or list): 
            - If `str`: A single prompt string representing the user's input to the LLM.
            - If `list`: A list of message dictionaries where each dictionary contains 
              a `role` ("human", "assistant", or "system") and `content` fields.
        system_prompt (str, optional): 
            - A system-level instruction to guide the behavior of the LLM. 
              Defaults to "You are an expert in generating anything. Your task will 
              be to follow the user instruction and generate accordingly."
        temp (float, optional): 
            - The temperature setting for the model. Controls randomness of output 
              (higher values increase randomness). Defaults to 0.2.
        model (str, optional): 
            - The LLM model to be used. Defaults to "gemini-1.5-flash."
        platform (str, optional): 
            - The platform hosting the LLM model. Defaults to "google."

    Returns:
        str: The generated text from the LLM based on the input prompts and configurations.

    Raises:
        ValueError: If `human_prompt` is neither a `str` nor a `list`.

    """
    
    if type(human_prompt)== str:
        messages = [
            {"role": "system", "content": system_prompt },
            {"role": "human", "content": human_prompt }
    ]
        # llm = ChatVertexAI(platform=platform, model=model, temperature=temp)
        output = get_llm().with_config({"run_name": "FUNC> LLM"}).invoke(messages).content
        return output
    elif type(human_prompt)== list:
        messages = [
            {"role": "system", "content": system_prompt }]
        messages += human_prompt
        # llm = ChatVertexAI(platform=platform, model=model,temperature=temp)
        output = get_llm().with_config({"run_name": "FUNC> LLM"}).invoke(messages).content
        return output
    
class GetSystemPromptArgs(BaseModel):
    query: str

class GetSystemPromptOutput(BaseModel):
    system_prompt: str

get_system_prompt_tool = Tool(
    name="get_system_prompt",
    func=get_system_prompt,
    description="Generates a system prompt instructing a specialist to complete a specific task based on a query.",
    args_schema=GetSystemPromptArgs,
    output_schema=GetSystemPromptOutput
)

class GetHumanPromptArgs(BaseModel):
    query: str

class GetHumanPromptOutput(BaseModel):
    human_prompt: str

get_human_prompt_tool = Tool(
    name="get_human_prompt",
    func=get_human_prompt,
    description="Generates a human prompt that provides a clear and specific task to be completed.",
    args_schema=GetHumanPromptArgs,
    output_schema=GetHumanPromptOutput
)

class QualityCheckerArgs(BaseModel):
    task: str
    output: str

class QualityCheckerOutput(BaseModel):
    suggestions: List[str]

quality_checker_tool = StructuredTool(
    name="quality_checker_bot",
    func=quality_checker_bot,
    description="Evaluates the quality of an output based on a given task and suggests improvements.",
    args_schema=QualityCheckerArgs,
    output_schema=QualityCheckerOutput
)

class GenerateContentArgs(BaseModel):
    human_prompt: str
    system_prompt: str = "You are an expert in generating anything. Your task will be to follow the user instruction and generate accordingly."
    temp: float = 0.2
    model: str = "gemini-1.5-flash"
    platform: str = "google"

class GenerateContentOutput(BaseModel):
    generated_text: str

generate_content_tool = StructuredTool(
    name="generate_content",
    func=generate_content,
    description="Generates text using a Large Language Model (LLM) with configurable parameters.",
    args_schema=GenerateContentArgs,
    output_schema=GenerateContentOutput
)

toolkit = [
    get_system_prompt_tool,
    get_human_prompt_tool,
    generate_content_tool
]

system_prompt = """ You are a smart AI assistant capable of generating system prompt, human prompt, using these prompts to generate content.
        Use your tools to answer questions. 
        If you do not have a tool to answer the question, say so. """

tool_calling_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

tool_runnable = create_tool_calling_agent(llm, toolkit, prompt  = tool_calling_prompt)

class AgentState(TypedDict):
    input: str
    agent_outcome: Union[AgentAction, list, ToolAgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[Union[tuple[AgentAction, str], tuple[ToolAgentAction, str]]], operator.add]

def run_tool_agent(state):
    # print("AGENT MODE")
    agent_outcome = tool_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}

tool_executor = ToolExecutor(toolkit)

def execute_tools(state):
    # print("ACTION MODE")
    agent_action = state['agent_outcome']
    if type(agent_action) is not list:
        agent_action = [agent_action]
    steps = []
    for action in agent_action:
        output = tool_executor.invoke(action)
        steps.append((action, str(output)))

    # print(f"OUTPUT: {output}")
    return {"intermediate_steps": steps}

def should_continue(data):
    if isinstance(data['agent_outcome'], AgentFinish):
        return "END"
    else:
        return "CONTINUE"
    
def define_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", run_tool_agent)
    workflow.add_node("action", execute_tools)
    workflow.set_entry_point("agent")
    workflow.add_edge('action', 'agent')
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "CONTINUE": "action",
            "END": END
        }
    )

    app = workflow.compile()
    return app

# Streamlit app
st.title("Content Generator Agent")

# Input query
query = st.text_input("Enter your query")


if st.button("Generate"):
    inputs = {"input": query}
    output = []
    app = define_workflow()

    for s in app.stream(inputs, config={"recursion_limit": 50}):
        temp_dict = {}
        out = list(s.values())[0]
        if list(out.keys())[0] == "agent_outcome":
            try:
                temp_dict["mode"] = "AGENT"
                temp_dict["output"] = f"{out['agent_outcome'][0].tool}"

                st.header("AGENT")
                st.write(out['agent_outcome'][0].tool)
            except:
                if out['agent_outcome'].return_values['output'] != "":
                    temp_dict["mode"] = "AGENT"
                    temp_dict["output"] = f"{out['agent_outcome'].return_values['output']}"
                    st.header("FINAL OUTPUT")
                    st.write(out['agent_outcome'].return_values['output'])
        elif list(out.keys())[0] == "intermediate_steps":
            temp_dict["mode"] = "ACTION"
            temp_dict["output"] = f"{out['intermediate_steps'][0][-1]}"
            st.header("ACTION")
            st.write(out['intermediate_steps'][0][-1])

        output.append(temp_dict)

