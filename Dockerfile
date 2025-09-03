FROM n8nio/n8n:1.101.0

# Cambiar al usuario adecuado (ya que 'root' no tiene acceso a node_modules de n8n)
USER root

# Instalar nodos adicionales
RUN npm install --prefix /home/node/.n8n \
    n8n-nodes-telegram-polling \
    n8n-nodes-base \
    @qdrant/qdrant-js \
    csv2xlsx

# Instalar pandoc usando apk (Alpine package manager)
RUN apk update && \
    apk add --no-cache pandoc curl unzip groff less python3 py3-pip

# Instalar AWS CLI v1 con override de seguridad
RUN pip3 install awscli --upgrade --break-system-packages


# Asegurarse de volver al usuario correcto
USER node
