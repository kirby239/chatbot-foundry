import os
import base64
import logging
import shutil
from fastapi import UploadFile, File, Form
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# --- IMPORTACIONES DE TELEMETRÍA ---
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from azure.core.settings import settings

# --- IMPORTACIONES DE AZURE ---
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
import azure.ai.projects.models as models
load_dotenv()

# --- SDK de Azure a usar el puente de OpenTelemetry ---
settings.tracing_implementation = "opentelemetry"
os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"

# --- CONFIGURACIÓN DE LANGFUSE ---
PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
HOST = os.getenv("LANGFUSE_BASE_URL")

# Crear el token de autenticación
auth_token = base64.b64encode(f"{PUBLIC_KEY}:{SECRET_KEY}".encode()).decode()

# Configurar el exportador a Langfuse
exporter = OTLPSpanExporter(
    endpoint=f"{HOST}/api/public/otel/v1/traces",
    headers={"Authorization": f"Basic {auth_token}"}
)

# Inicializar el Tracer
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# --- APP FASTAPI Y AZURE FOUNDRY ---
app = FastAPI(title="Azure AI Foundry Agent API")

# --- CONFIGURACIÓN DE Azure Foundry ---
PROJECT_ENDPOINT = os.getenv("AZURE_AI_FOUNDRY_ENDPOINT")
MODEL_DEPLOYMENT = os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini")
credential = DefaultAzureCredential()
project_client = AIProjectClient(endpoint=PROJECT_ENDPOINT, credential=credential, enable_tracing=True)

class AgentRequest(BaseModel):
    name: str
    instructions: str
    file: UploadFile = File(None)

class PromptRequest(BaseModel):
    prompt: str

# --- ENDPOINTS ---

# @app.post("/agents", summary="Crear un nuevo agente")
# async def create_agent(req: AgentRequest):
#     with tracer.start_as_current_span("Foundry_Create_Agent") as span:
#         try:
#             tools = []
#             tool_resources = None

#             # SI EL USUARIO MANDA UN ARCHIVO, CONFIGURAMOS EL CONOCIMIENTO
#             if req.file_path:
#                 print("file" , req.file_path)
                
#                 if not os.path.exists(req.file_path):
#                     raise HTTPException(status_code=400, detail=f"Archivo no encontrado: {req.file_path}")
                
#                 # 1. Subir el archivo .md
#                 uploaded_file = project_client.agents.files.upload(
#                     file_path=req.file_path,
#                     purpose="assistants"
#                 )
#                 print("uploaded_file",uploaded_file)

#                 # 2. Crear el Vector Store para que el agente pueda "leer" el contenido
#                 vector_store = project_client.agents.vector_stores.create(
#                     file_ids=[uploaded_file.id], 
#                     name=f"vs_{req.name}"
#                 )
#                 print("vector_store", vector_store)
#                 # 3. Preparar la herramienta de búsqueda
#                 try:
#                     from azure.ai.projects.models import FileSearchToolDefinition, ToolResources, FileSearchToolResource
#                 except ImportError:
#                     from azure.ai.agents.models import FileSearchToolDefinition, ToolResources, FileSearchToolResource

#                 tools = [FileSearchToolDefinition()]
#                 tool_resources = ToolResources(
#                     file_search=FileSearchToolResource(vector_store_ids=[vector_store.id])
#                 )
#             # CREACIÓN DEL AGENTE (con o sin herramientas dependiendo del 'if')

#             agent = project_client.agents.create_agent(
#                 model=MODEL_DEPLOYMENT,
#                 name=req.name,
#                 instructions=req.instructions,
#                 tools=tools,
#                 tool_resources=tool_resources
#             )
#             return {"id": agent.id, "name": agent.name, "instrucciones":agent.instructions,"has_knowledge": req.file_path is not None, "status": "created"}
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents", summary="Crear un nuevo agente con archivo automático")
async def create_agent(
    name: str = Form(...),              # Ahora usamos Form porque enviamos archivos
    instructions: str = Form(...),
    file: UploadFile = File(None)       # El archivo real que viene de Postman
):
    with tracer.start_as_current_span("Foundry_Create_Agent") as span:
        temp_path = None
        try:
            tools = []
            tool_resources = None

            # 1. Si el usuario envió un archivo, lo guardamos localmente en el repo
            if file:
                temp_path = os.path.join(os.getcwd(), file.filename)
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                print(f"✅ Archivo guardado temporalmente en el repo: {temp_path}")

                # 2. Subir a Azure AI Foundry
                uploaded_file = project_client.agents.files.upload(
                    file_path=temp_path,
                    purpose="assistants"
                )

                # 3. Crear Vector Store
                vector_store = project_client.agents.vector_stores.create(
                    file_ids=[uploaded_file.id], 
                    name=f"vs_{name}"
                )

                # 4. Importar clases oficiales (Versión 1.0.0+)
                try:
                     from azure.ai.projects.models import FileSearchToolDefinition, ToolResources, FileSearchToolResource
                except ImportError:
                     from azure.ai.agents.models import FileSearchToolDefinition, ToolResources, FileSearchToolResource
                tools = [FileSearchToolDefinition()]
                tool_resources = ToolResources(
                     file_search=FileSearchToolResource(vector_store_ids=[vector_store.id])
                 )

            # 5. CREACIÓN DEL AGENTE
            agent = project_client.agents.create_agent(
                model=MODEL_DEPLOYMENT,
                name=name,
                instructions=instructions,
                tools=tools,
                tool_resources=tool_resources
            )

            # 6. ÉXITO (200 OK): Borramos el archivo del repo
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"🗑️ Proceso exitoso: Archivo {file.filename} eliminado del repo.")

            return {
                "id": agent.id, 
                "name": agent.name, 
                "status": "created",
                "file_processed": file.filename if file else None
            }

        except Exception as e:
            # En caso de error, también intentamos limpiar para no dejar basura
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/list-agents", summary="Listar todos los agentes")
async def list_agents():
    try:
        agents = project_client.agents.list_agents()
        return [{"id": a.id, "name": a.name} for a in agents]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/{agent_id}/prompt")
async def send_prompt(agent_id: str, req: PromptRequest):
    # Envolvemos la ejecución en un span principal para Langfuse
    with tracer.start_as_current_span("Foundry_Agent_Call") as span:
        span.set_attribute("gen_ai.response.model", "gpt-4.1-mini-2025-04-14")
        span.set_attribute("model", "gpt-4.1-mini-2025-04-14")
        try:
            span.set_attribute("agent_id", agent_id)
            
            thread = project_client.agents.threads.create()
            project_client.agents.messages.create(
                thread_id=thread.id, 
                role="user", 
                content=req.prompt
            )

            run = project_client.agents.runs.create_and_process(
                thread_id=thread.id,
                agent_id=agent_id
            )
            
            if str(run.status).lower().endswith("completed"):
                # --- AQUÍ CAPTURAMOS LOS TOKENS ---
                if hasattr(run, 'usage') and run.usage:
                    span.set_attribute("gen_ai.usage.input_tokens", run.usage.prompt_tokens)
                    span.set_attribute("gen_ai.usage.output_tokens", run.usage.completion_tokens)
                    
                messages = project_client.agents.messages.list(thread_id=thread.id)
                for msg in messages:
                    if msg.role == "assistant":
                        last_text_value = msg.content[0].text.value

                        # Forzamos el envío de datos antes de terminar
                        provider.force_flush()
                        
                        return {
                            "response": last_text_value,
                            "agent_id": agent_id,
                            "thread_id": thread.id
                        }
            provider.force_flush()
            return {"status": str(run.status), "agent_id": agent_id, "detail": "No completado"}
            
        except Exception as e:
            span.record_exception(e)
            provider.force_flush()
            raise HTTPException(status_code=500, detail=str(e))

# Prueba de conexión al arrancar
@app.on_event("startup")
async def startup_event():
    with tracer.start_as_current_span("Prueba_Inicio_App"):
        print("🚀 App iniciada. Enviando traza de prueba a Langfuse...")