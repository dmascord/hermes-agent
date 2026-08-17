// Hermes API auth sidecar.
//
// Traefik calls /validate before forwarding a request to Hermes. The sidecar
// validates the caller's Bearer token against the configured key registry and
// returns an internal credential that Traefik forwards to Hermes.
package main

import (
	"crypto/subtle"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

type keyRecord struct {
	ID     string `json:"id"`
	Secret string `json:"secret"`
}

type keyRegistry struct {
	Keys []keyRecord `json:"keys"`
}

type server struct {
	keys          []keyRecord
	internalToken string
	logger        *slog.Logger
}

func loadRegistry() ([]keyRecord, error) {
	raw := strings.TrimSpace(os.Getenv("HERMES_API_KEYS_JSON"))
	if raw == "" {
		return nil, errors.New("HERMES_API_KEYS_JSON is required")
	}

	var registry keyRegistry
	if err := json.Unmarshal([]byte(raw), &registry); err != nil {
		return nil, errors.New("HERMES_API_KEYS_JSON must be valid JSON")
	}

	seenIDs := make(map[string]struct{}, len(registry.Keys))
	keys := make([]keyRecord, 0, len(registry.Keys))
	for _, key := range registry.Keys {
		key.ID = strings.TrimSpace(key.ID)
		key.Secret = strings.TrimSpace(key.Secret)
		if key.ID == "" || key.Secret == "" {
			return nil, errors.New("every API key requires non-empty id and secret fields")
		}
		if _, exists := seenIDs[key.ID]; exists {
			return nil, errors.New("API key ids must be unique")
		}
		seenIDs[key.ID] = struct{}{}
		keys = append(keys, key)
	}
	if len(keys) == 0 {
		return nil, errors.New("at least one API key is required")
	}
	return keys, nil
}

func bearerToken(header string) (string, bool) {
	const prefix = "Bearer "
	if !strings.HasPrefix(header, prefix) {
		return "", false
	}
	token := strings.TrimSpace(strings.TrimPrefix(header, prefix))
	return token, token != ""
}

func (s *server) validate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	token, ok := bearerToken(r.Header.Get("Authorization"))
	if !ok {
		s.reject(w, r, "missing_or_malformed_authorization")
		return
	}

	matchedID := ""
	for _, key := range s.keys {
		if subtle.ConstantTimeCompare([]byte(token), []byte(key.Secret)) == 1 {
			matchedID = key.ID
		}
	}
	if matchedID == "" {
		s.reject(w, r, "invalid_key")
		return
	}

	// Traefik copies these response headers onto the request sent to Hermes.
	w.Header().Set("Authorization", "Bearer "+s.internalToken)
	w.Header().Set("X-Hermes-Consumer", matchedID)
	w.WriteHeader(http.StatusOK)
	s.logger.Info("request authorized",
		"consumer", matchedID,
		"method", r.Header.Get("X-Forwarded-Method"),
		"uri", r.Header.Get("X-Forwarded-Uri"),
		"remote", r.Header.Get("X-Forwarded-For"),
	)
}

func (s *server) reject(w http.ResponseWriter, r *http.Request, reason string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusUnauthorized)
	_, _ = w.Write([]byte(`{"error":{"message":"Invalid API key","type":"invalid_request_error","code":"invalid_api_key"}}`))
	s.logger.Warn("request rejected",
		"reason", reason,
		"method", r.Header.Get("X-Forwarded-Method"),
		"uri", r.Header.Get("X-Forwarded-Uri"),
		"remote", r.Header.Get("X-Forwarded-For"),
	)
}

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	keys, err := loadRegistry()
	if err != nil {
		logger.Error("invalid key registry", "error", err)
		os.Exit(1)
	}
	internalToken := strings.TrimSpace(os.Getenv("HERMES_INTERNAL_TOKEN"))
	if len(internalToken) < 32 {
		logger.Error("HERMES_INTERNAL_TOKEN must contain at least 32 characters")
		os.Exit(1)
	}

	s := &server{keys: keys, internalToken: internalToken, logger: logger}
	mux := http.NewServeMux()
	mux.HandleFunc("/validate", s.validate)
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	httpServer := &http.Server{
		Addr:              ":8081",
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-stop
		_ = httpServer.Close()
	}()

	logger.Info("auth sidecar listening", "address", httpServer.Addr, "key_count", len(keys))
	if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		logger.Error("auth sidecar failed", "error", err)
		os.Exit(1)
	}
}
