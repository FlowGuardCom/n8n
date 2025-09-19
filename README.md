# 🚀 Proyecto n8n + Ollama + Qdrant

Este proyecto despliega un entorno completo de **automatización (n8n)**, **embeddings/LLM (Ollama)** y **vector database (Qdrant)** usando **Docker Compose**. Está preparado para ejecutarse en máquinas con GPU y soporta la restauración de flujos y credenciales de n8n mediante una base de datos `sqlite`.

---

## 📋 Requisitos de la máquina

1. **Sistema operativo**: Ubuntu 22.04 LTS (o equivalente).
2. **Usuario con acceso SSH (y sudo si fuera posible).**
3. **Paquetes básicos**:
```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y git curl wget unzip htop build-essential
```
4. **Docker + Docker Compose**:
```bash
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
    sudo apt install -y docker-compose-plugin
```
5. **Soporte GPU (opcional, recomendado para Ollama)**:
```bash
    sudo apt install -y nvidia-utils-525 nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
```
6. **Red y puertos**:  
Por defecto, el `docker-compose.yml` expone estos servicios en el host:  

| Servicio | Puerto host | Descripción                       |  
|----------|-------------|-----------------------------------|  
| n8n      | 5678        | Panel web de n8n                  |  
| Ollama   | 11434       | API REST para modelos LLM/embeds  |  
| Qdrant   | 6333        | API REST/gRPC de la vector DB     |  
   
## 📦 Clonar el proyecto
Clonar proyecto n8n en GitHub
```bash
    git clone https://github.com/FlowGuardCom/n8n.git
    cd n8n
```
    
## 🐳 Desplegar servicios
El stack incluye:
- n8n: automatización y flujos.
- Ollama: modelos LLM y embeddings.
- Qdrant: base vectorial para almacenamiento y búsqueda.
```bash
    docker compose up --build -d
```
Verificar:
- n8n: http://IP_PUBLICA:5678
- Ollama: http://IP_PUBLICA:11434
- Qdrant: http://IP_PUBLICA:6333

## 🤖 Descargar modelos Ollama
Una vez desplegado:
```bash
    docker exec -it ollama ollama pull llama3.1
    docker exec -it ollama ollama pull nomic-embed-text
```

## 🔄 Migración de flujos y credenciales de n8n
Los flujos y credenciales de n8n están en n8n_data/database.sqlite.
1. Transferir archivo vía SSH/SCP
```bash
    scp -i ~/.ssh/clave.pem <PATH>/database_YYYY-MM-DD.sqlite.gz user@IP_DESTINO:/home/ec2-user/
```
2. Restaurar en la máquina destino
```bash
    mv /home/user/database_YYYY-MM-DD.sqlite.gz ./n8n_data/
    gunzip ./n8n_data/database_YYYY-MM-DD.sqlite.gz
    mv ./n8n_data/database_YYYY-MM-DD.sqlite ./n8n_data/database.sqlite
    docker compose restart n8n
```

## 🔐 Conexión SSH a la máquina
```bash
    ssh -i ~/.ssh/clave.pem ubuntu@IP_PUBLICA
```

## Subida de documentos al sistema RAG
Una vez desplegado, entrar en la siguiente url (se encuentra en el flujo "RAG System", en el nodo trigger "Upload file"):
```bash
    http://3.77.214.209:5678/form/233e6641-8e83-4409-a108-602ab471b569
```

Credenciales:
- usuario: airtrace
- contraseña: airtrace

## Consulta a sistema RAG
Es necesario tener activo el flujo "Agente Consulta RAG" y en él, tener activado el nodo "webhook"
y desactivado el "When chat message received". La url de laa consulta es la siguiente:
```bash
    http://localhost:5678/webhook-test/consulta-rag
```

Ejemplo de consulta:
```bash
    http://localhost:5678/webhook-test/consulta-rag
```




