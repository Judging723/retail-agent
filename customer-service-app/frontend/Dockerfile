FROM node:24-alpine AS build

WORKDIR /app
COPY customer-service-app/frontend/package.json customer-service-app/frontend/package-lock.json ./
RUN npm ci
COPY customer-service-app/frontend ./
RUN npm run build

FROM nginx:1.27-alpine
COPY customer-service-app/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
