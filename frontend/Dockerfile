# ==============================================================================
# Clizzer Frontend Dockerfile
# Node 20 Alpine for Next.js GTA-Themed UI
# ==============================================================================

FROM node:20-alpine

# Set working directory
WORKDIR /app

# Set environment variables
ENV NODE_ENV=production \
    PORT=3000

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy application source code
COPY . .

# Build production assets
RUN npm run build

# Expose Next.js server port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD wget -qO- http://localhost:3000 || exit 1

# Start the Next.js production server
CMD ["npm", "start"]
