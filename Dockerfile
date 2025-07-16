FROM n8nio/n8n:1.101.0

# Cambiar al usuario adecuado (ya que 'root' no tiene acceso a node_modules de n8n)
USER root

# Instalar nodos adicionales
RUN npm install --prefix /home/node/.n8n \
    n8n-nodes-telegram-polling \
    n8n-nodes-base

# Instalar pandoc usando apk (Alpine package manager)
RUN apk update && \
    apk add pandoc

RUN npm install -g @qdrant/qdrant-js

# Asegurarse de volver al usuario correcto
USER node
