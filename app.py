import os
from fastapi import FastAPI, HTTPException
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Azure AI Foundry Agent API")

PROJECT_ENDPOINT = os.getenv("AZURE_AI_FOUNDRY_ENDPOINT")
MODEL_DEPLOYMENT = os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini")
credential = DefaultAzureCredential()
project_client = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=credential)

class AgentRequest(BaseModel):
    name: str
    instructions: str

class PromptRequest(BaseModel):
    prompt: str

# --- ENDPOINTS ---

@app.post("/agents", summary="Crear un nuevo agente")
async def create_agent(req: AgentRequest):
    try:
        agent = project_client.agents.create_agent(
            model=MODEL_DEPLOYMENT,
            name=req.name,
            instructions=req.instructions
        )
        return {"id": agent.id, "name": agent.name, "instrucciones":agent.instructions, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents", summary="Listar todos los agentes")
async def list_agents():
    try:
        # CORRECCIÓN 1: Se elimina .data porque 'agents' es un iterable directo
        agents = project_client.agents.list_agents()
        return [{"id": a.id, "name": a.name} for a in agents]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/{agent_id}/prompt", summary="Enviar mensaje a un agente")
async def send_prompt(agent_id: str, req: PromptRequest):
    try:
        thread = project_client.agents.create_thread()
        
        project_client.agents.create_message(
            thread_id=thread.id, 
            role="user", 
            content=req.prompt
        )
        
        run = project_client.agents.create_and_poll_run(
            assistant_id=agent_id, 
            thread_id=thread.id
        )
        
        if run.status == "completed":
            # CORRECCIÓN 2: Se elimina .data para acceder a los mensajes
            messages = project_client.agents.list_messages(thread_id=thread.id)
            
            # Convertimos a lista para acceder al primer elemento [0]
            msg_list = list(messages)
            agent_response = msg_list[0].content[0].text.value
            return {"agent_id": agent_id, "response": agent_response}
        else:
            return {"status": run.status, "detail": "El agente no pudo completar la tarea"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))