FROM oven/bun:1 AS builder

WORKDIR /app
COPY package.json bun.lock ./
RUN bun install

COPY . .
RUN bun run build

FROM node:22-slim
WORKDIR /app
COPY --from=builder /app/package.json .

RUN npm install -g @sveltejs/kit

EXPOSE 3000
CMD ["node", "build"]
