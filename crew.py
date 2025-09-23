import os
import pickle
import faiss
import numpy as np
import pandas as pd
import docx
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pydantic import BaseModel
from typing import List
import google.generativeai as genai
from new1 import query_vector_db as data_rag
from new import query_vector_db1 as temp_rag
from crewai import Agent, Task, Process, Crew, LLM,CrewOutput
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# =========================
# ENV + API KEYS
# =========================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
web_search = SerperDevTool()
# =========================
# LLM
# =========================
llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.7,
    api_key=os.environ["GOOGLE_API_KEY"],
)

# =========================
# KNOWLEDGE SOURCES
# =========================
class SourceMapping(BaseModel):
    source: str = Field(..., description="Reference source for this section or subsection.")
    note: Optional[str] = Field(None, description="Additional context about how the source was used.")

class WinThemeAlignment(BaseModel):
    theme: str = Field(..., description="The win theme name.")
    note: str = Field(..., description="Explanation of how this theme is addressed.")

class SubSection(BaseModel):
    subsection_title: Optional[str] = Field(None, description="Title of the subsection.")
    subsection_requirement: Optional[str] = Field(None, description="Requirement description for this subsection.")
    context: Optional[str] = Field(None, description="Contextual notes for subsection.")
    subsection_purpose: Optional[str] = Field(None, description="Purpose of this subsection.")
    subsection_instructions_to_writer: Optional[str] = Field(None, description="Instructions for the writer.")
    subsection_content: Optional[str] = Field(None, description="Generated or drafted content for this subsection.")
    subsection_source_mapping: List[SourceMapping] = Field(default_factory=list, description="Mapping to original RFP sources.")
    subsection_win_theme_alignment: List[WinThemeAlignment] = Field(default_factory=list, description="How subsection aligns with win themes.")

class Section(BaseModel):
    solicitation_id: str = Field(..., description="Unique solicitation identifier.")
    section_title: str = Field(..., description="Title of the section.")
    section_purpose: str = Field(..., description="Purpose of this section.")
    section_instructions_to_writer: Optional[str] = Field(None, description="Guidelines for the writer on this section.")
    section_content: Optional[str] = Field(None, description="Generated content for the section.")
    section_source_mapping: List[SourceMapping] = Field(default_factory=list, description="Mapping to original RFP sources.")
    section_win_theme_alignment: List[WinThemeAlignment] = Field(default_factory=list, description="How section aligns with win themes.")
    subsections: List[SubSection] = Field(default_factory=list, description="Subsections under this section.")
    refinement_prompt: Optional[str] = Field(None, description="Prompt to refine this section.")

class ProposalOutput(BaseModel):
    proposal: List[Section] = Field(..., description="List of proposal sections with details.")




Manager=Agent(
  role="the project manager",
  goal="To manage the entire proposal generation process from end to end, ensuring all agents work on the correct data and produce high-quality, compliant outputs. It's the ultimate gatekeeper for quality and integrity. And also make sure that none of agent should make a empty llm calls.",
  backstory="You are a seasoned veteran of countless high-stakes proposal battles. You've seen it all: last-minute crises, missing data, and fierce competition. Your reputation was built on a single principle: ruthless attention to detail and a zero-tolerance policy for non-compliance. You've learned that every successful proposal is built on a solid foundation of verifiable facts, not wishful thinking. Your singular purpose now is to ensure the integrity of the process and deliver a final product that is not just good, but unflinchingly perfect and compliant.",
  llm=llm,
  max_rpm=5
)
# =========================
# FAISS + EMBEDDING HELPERS
# =========================
INDEX_PATH = "./vectorstore/faiss_index"
DIMENSION = 500
EMBEDDING_MODEL = "models/embedding-001"


# =========================
# CREW
# =========================


@CrewBase
class ProposalCrew:
    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"



    # === Agents ===
    @agent
    def Data_Ingestion_Agent(self) -> Agent:
        return Agent(**self.agents_config["Data_Ingestion_Agent"], tools=[data_rag,temp_rag,web_search], llm=llm,max_iter=10,max_rpm=9)
    @agent
    def Executive_Summary_Agent(self) -> Agent:
        return Agent(**self.agents_config["Executive_Summary_Agent"], tools=[data_rag,temp_rag,web_search], llm=llm,max_iter=5,max_rpm=9)
    @agent
    def Corporate_Data_Extractor(self) -> Agent:
        return Agent(**self.agents_config["Corporate_Data_Extractor"], tools=[data_rag],llm=llm,max_iter=5,max_rpm=9)
    @agent
    def Technical_Approach_Agent(self) -> Agent:
        return Agent(**self.agents_config["Technical_Approach_Agent"], tools=[data_rag,temp_rag,web_search], llm=llm,max_iter=5,max_rpm=9)
    @agent
    def Price_Section_Agent(self)->Agent:
        return Agent(**self.agents_config["Price_Section_Agent"],tools=[data_rag,temp_rag,web_search],llm=llm,max_iter=5,max_rpm=9)
    @agent
    def Past_Performance_Agent(self) -> Agent:
        return Agent(**self.agents_config["Past_Performance_Agent"],tools=[data_rag,temp_rag,web_search],llm=llm,max_iter=5,max_rpm=9)
    @agent
    def Management_Section_Agent(self) -> Agent:
        return Agent(**self.agents_config["Management_Section_Agent"],tools=[data_rag,temp_rag,web_search],llm=llm,max_iter=5,max_rpm=9)
    @agent
    def Other_Extra_Data_Agent(self) -> Agent:
        return Agent(**self.agents_config["Other_Extra_Data_Agent"],tools=[data_rag,temp_rag,web_search],llm=llm,max_iter=5,max_rpm=9)
    @agent
    def Data_Compliance_Agent(self) -> Agent:
        return Agent(**self.agents_config["Data_Compliance_Agent"],llm=llm,max_rpm=9)
    # === Tasks ===
    @task
    def RFP_and_Data_Intake(self) -> Task:
        return Task(**self.tasks_config["RFP_and_Data_Intake"])

    @task
    def Requirements_Audit_Gap_Analysis(self) -> Task:
        return Task(**self.tasks_config["Requirements_Audit_Gap_Analysis"])

    @task
    def Initial_Delegation(self) -> Task:
        return Task(**self.tasks_config["Initial_Delegation"])
    @task
    def Quality_Control_and_Iteration(self)-> Task:
        return Task(**self.tasks_config["Quality_Control_and_Iteration"])
    
    @task
    def Final_Assembly(self) -> Task:
        return Task(**self.tasks_config["Final_Assembly"])
    
    @task
    def Final_formatting(self) -> Task:
        return Task(**self.tasks_config["Final_formatting"],output_json=ProposalOutput)
    
    # === Crew ===
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            verbose=True,
            max_rpm=9,
            manager_agent=Manager,
            output_json=ProposalOutput,
            output_log_file="my_crew_logs.json",
            allow_delegation=True,
            embedder={
                "provider": "google",
                "config": {
                    "api_key": os.environ["GOOGLE_API_KEY"],
                    "model": "gemini-embedding-001",
                },
            },
        )
