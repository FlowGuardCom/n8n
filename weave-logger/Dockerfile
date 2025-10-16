FROM node:20-slim

# Crear directorio de trabajo
WORKDIR /app

# Copiar package.json y lock (si existe)
COPY package*.json ./
RUN npm install --omit=dev

# Copiar el código
COPY . .

# Exponer puerto
EXPOSE 3000

# Comando por defecto
CMD ["npm", "start"]
