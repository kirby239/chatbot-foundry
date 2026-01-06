import os
import logging
from fastapi import FastAPI, HTTPException
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from pydantic import BaseModel
from dotenv import load_dotenv
from azure.monitor.opentelemetry import configure_azure_monitor
load_dotenv()
#crear en el  Application Insights de Azure y copiar la Connection string
conn_string = os.getenv("APPLICATION_INSIGHTS_CONNECTION_STRING")

configure_azure_monitor(connection_string=conn_string)

# AÑADE ESTO PARA TESTEAR
logger = logging.getLogger("azure.monitor.test")
logger.setLevel(logging.INFO)
print("Enviando señal de vida a Azure Monitor...")
logger.info("Iniciando monitoreo del Agente IA")

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
        # 1. Crear el hilo
        thread = project_client.agents.threads.create()
        print(f"Thread creado con ID: {thread.id}")
        
        # 2. Crear el mensaje
        message = project_client.agents.messages.create(
            thread_id=thread.id, 
            role="user", 
            content=req.prompt
        )
        # Nota: message suele ser un objeto, si falla como diccionario usa message.id
        msg_id = message.id if hasattr(message, 'id') else message.get('id')
        print(f"Mensaje creado con ID: {msg_id}")

        # 3. Ejecutar y esperar
        run = project_client.agents.runs.create_and_process(
            thread_id=thread.id,
            agent_id=agent_id
        )
        
        # En tu consola sale RunStatus.COMPLETED, lo comparamos correctamente
        print(f"Ejecución iniciada con ID: {run.id}, estado actual: {run.status}")
        
        if run.status == "completed" or str(run.status).lower().endswith("completed"):
            # 4. Listar mensajes
            messages = project_client.agents.messages.list(thread_id=thread.id)
            
            for msg in messages:
                # Solo procesamos la respuesta del asistente
                if msg.role == "assistant":
                    print(f"Mensaje recibido con ID: {msg.id}")
                    
                    # CORRECCIÓN AQUÍ: Acceso correcto al valor del texto
                    # El error decía que 'MessageTextContent' no tiene 'value' 
                    # porque el valor está en msg.content[0].text.value
                    try:
                        last_text_value = msg.content[0].text.value
                        print(f"Respuesta del agente: {last_text_value}")
                        
                        return {
                            "agent_id": agent_id, 
                            "thread_id": thread.id,
                            "response": last_text_value
                        }
                    except (IndexError, AttributeError) as e:
                        print(f"Error accediendo al contenido: {e}")
                        continue
            
            return {"detail": "No se encontró contenido de texto en la respuesta"}
            
        else:
            return {
                "status": str(run.status), 
                "agent_id": agent_id,
                "detail": f"Estado no completado: {run.status}"
            }
            
    except Exception as e:
        print(f"Error técnico real: {type(e).__name__} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        #agents.delete_aget(shrmy-id) elimina el agente
        #agents.thread