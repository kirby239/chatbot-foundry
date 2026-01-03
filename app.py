import os
from fastapi import FastAPI, HTTPException
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from pydantic import BaseModel
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

app = FastAPI(title="Azure AI Foundry Agent API")

# Configuración
PROJECT_ENDPOINT = os.getenv("AZURE_AI_FOUNDRY_ENDPOINT")
MODEL_DEPLOYMENT = os.getenv("MODEL_DEPLOYMENT_NAME")

# Inicializar cliente de Azure
# DefaultAzureCredential usará tu login de 'az login' en local 
# o el Service Principal en GitHub Actions.
credential = DefaultAzureCredential()
project_client = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=credential)

# Modelos de datos para las peticiones
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
        return {"id": agent.id, "name": agent.name, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents", summary="Listar todos los agentes")
async def list_agents():
    try:
        agents = project_client.agents.list_agents()
        return [{"id": a.id, "name": a.name} for a in agents.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/{agent_id}/prompt", summary="Enviar mensaje a un agente")
async def send_prompt(agent_id: str, req: PromptRequest):
    try:
        # 1. Crear un hilo de conversación (Thread)
        thread = project_client.agents.create_thread()
        
        # 2. Crear el mensaje del usuario
        project_client.agents.create_message(
            thread_id=thread.id, 
            role="user", 
            content=req.prompt
        )
        
        # 3. Ejecutar el agente y esperar respuesta (Polling)
        run = project_client.agents.create_and_poll_run(
            assistant_id=agent_id, 
            thread_id=thread.id
        )
        
        if run.status == "completed":
            # 4. Recuperar los mensajes del hilo
            messages = project_client.agents.list_messages(thread_id=thread.id)
            # El primer mensaje en la lista suele ser la respuesta más reciente
            agent_response = messages.data[0].content[0].text.value
            return {"agent_id": agent_id, "response": agent_response}
        else:
            return {"status": run.status, "detail": "El agente no pudo completar la tarea"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))