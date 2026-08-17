# syntax=docker/dockerfile:1
FROM golang:1.24-alpine AS build
WORKDIR /src
COPY main.go ./
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags='-s -w' -o /out/hermes-api-auth ./main.go

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/hermes-api-auth /hermes-api-auth
EXPOSE 8081
USER nonroot:nonroot
ENTRYPOINT ["/hermes-api-auth"]
