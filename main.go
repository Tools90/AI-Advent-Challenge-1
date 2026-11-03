package main

import (
	"bufio"
	"bytes"
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

//go:embed web/index.html
var webFS embed.FS

type config struct {
	apiKey  string
	baseURL string
	model   string
	addr    string
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Messages []chatMessage `json:"messages"`
}

type chatResponse struct {
	Reply string `json:"reply"`
	Model string `json:"model"`
}

type apiError struct {
	Status  int
	Message string
}

func (e *apiError) Error() string { return e.Message }

type server struct {
	cfg    config
	client *http.Client
}

func main() {
	if err := loadDotEnv(".env"); err != nil {
		log.Printf("config: %v", err)
	}

	cfg, err := loadConfig()
	if err != nil {
		log.Fatalf("config: %v", err)
	}

	s := &server{
		cfg:    cfg,
		client: &http.Client{Timeout: 120 * time.Second},
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /", s.index)
	mux.HandleFunc("POST /api/chat", s.chat)
	mux.HandleFunc("GET /healthz", s.health)

	srv := &http.Server{
		Addr:              cfg.addr,
		Handler:           logRequests(mux),
		ReadHeaderTimeout: 10 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		log.Printf("сервер запущен: http://localhost%s (модель: %s)", cfg.addr, cfg.model)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("server: %v", err)
		}
	}()

	<-ctx.Done()
	log.Println("останавливаюсь...")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("shutdown: %v", err)
	}
}

func loadConfig() (config, error) {
	apiKey, err := loadAPIKey()
	if err != nil {
		return config{}, err
	}

	baseURL := strings.TrimRight(strings.TrimSpace(os.Getenv("DEEPSEEK_BASE_URL")), "/")
	if baseURL == "" {
		baseURL = "https://api.deepseek.com"
	}

	model := strings.TrimSpace(os.Getenv("DEEPSEEK_MODEL"))
	if model == "" {
		model = "deepseek-chat"
	}

	addr := strings.TrimSpace(os.Getenv("ADDR"))
	if port := strings.TrimSpace(os.Getenv("PORT")); addr == "" && port != "" {
		addr = ":" + port
	}
	if addr == "" {
		addr = ":8080"
	}

	return config{apiKey: apiKey, baseURL: baseURL, model: model, addr: addr}, nil
}

const placeholderKey = "sk-your-key-here"

func loadAPIKey() (string, error) {
	if key := strings.TrimSpace(os.Getenv("DEEPSEEK_API_KEY")); key != "" && key != placeholderKey {
		return key, nil
	}

	path := strings.TrimSpace(os.Getenv("DEEPSEEK_API_KEY_FILE"))
	if path == "" {
		path = "api_key.txt"
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", fmt.Errorf("ключ API не найден: создайте файл %s и вставьте в него ключ DeepSeek", path)
		}
		return "", fmt.Errorf("чтение %s: %w", path, err)
	}

	key := unquote(strings.TrimSpace(strings.TrimPrefix(string(data), "\ufeff")))
	if key == "" || key == placeholderKey {
		return "", fmt.Errorf("файл %s пуст: вставьте в него ключ DeepSeek", path)
	}
	return key, nil
}

func (s *server) index(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}

	data, err := webFS.ReadFile("web/index.html")
	if err != nil {
		http.Error(w, "index.html не найден", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(data)
}

func (s *server) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "model": s.cfg.model})
}

func (s *server) chat(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)

	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "Некорректный JSON: "+err.Error())
		return
	}

	if err := validateMessages(req.Messages); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	reply, model, err := s.askDeepSeek(r.Context(), req.Messages)
	if err != nil {
		var ae *apiError
		if errors.As(err, &ae) {
			writeError(w, ae.Status, ae.Message)
			return
		}
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}

	writeJSON(w, http.StatusOK, chatResponse{Reply: reply, Model: model})
}

func (s *server) askDeepSeek(ctx context.Context, messages []chatMessage) (string, string, error) {
	payload, err := json.Marshal(map[string]any{
		"model":    s.cfg.model,
		"messages": messages,
		"stream":   false,
	})
	if err != nil {
		return "", "", fmt.Errorf("сборка запроса: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.baseURL+"/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return "", "", fmt.Errorf("создание запроса: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+s.cfg.apiKey)

	resp, err := s.client.Do(req)
	if err != nil {
		return "", "", fmt.Errorf("запрос к DeepSeek не удался: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	if err != nil {
		return "", "", fmt.Errorf("чтение ответа DeepSeek: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var apiErr struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if json.Unmarshal(body, &apiErr) == nil && apiErr.Error.Message != "" {
			return "", "", &apiError{Status: resp.StatusCode, Message: "DeepSeek: " + apiErr.Error.Message}
		}
		return "", "", &apiError{Status: resp.StatusCode, Message: fmt.Sprintf("DeepSeek вернул статус %d", resp.StatusCode)}
	}

	var out struct {
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		return "", "", fmt.Errorf("разбор ответа DeepSeek: %w", err)
	}
	if len(out.Choices) == 0 || strings.TrimSpace(out.Choices[0].Message.Content) == "" {
		return "", "", errors.New("DeepSeek вернул пустой ответ")
	}

	model := out.Model
	if model == "" {
		model = s.cfg.model
	}
	return out.Choices[0].Message.Content, model, nil
}

func validateMessages(messages []chatMessage) error {
	if len(messages) == 0 {
		return errors.New("список сообщений пуст")
	}
	if len(messages) > 200 {
		return errors.New("слишком много сообщений (максимум 200)")
	}

	total := 0
	for i, m := range messages {
		switch m.Role {
		case "system", "user", "assistant":
		default:
			return fmt.Errorf("сообщение #%d: недопустимая роль %q", i+1, m.Role)
		}
		if strings.TrimSpace(m.Content) == "" {
			return fmt.Errorf("сообщение #%d: пустой текст", i+1)
		}
		total += len(m.Content)
	}
	if total > 200_000 {
		return errors.New("слишком длинная переписка (максимум 200 000 символов)")
	}
	return nil
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start).Round(time.Millisecond))
	})
}

func loadDotEnv(path string) error {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("файл %s не найден, использую переменные окружения", path)
		}
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(strings.TrimPrefix(scanner.Text(), "\ufeff"))
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(strings.TrimPrefix(key, "export "))
		value = unquote(strings.TrimSpace(value))
		if key == "" {
			continue
		}

		if _, exists := os.LookupEnv(key); !exists {
			_ = os.Setenv(key, value)
		}
	}
	return scanner.Err()
}

func unquote(v string) string {
	if len(v) >= 2 {
		if (v[0] == '"' && v[len(v)-1] == '"') || (v[0] == '\'' && v[len(v)-1] == '\'') {
			return v[1 : len(v)-1]
		}
	}
	return v
}
