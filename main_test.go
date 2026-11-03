package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func newTestServer(t *testing.T, upstream http.HandlerFunc) *server {
	t.Helper()

	up := httptest.NewServer(upstream)
	t.Cleanup(up.Close)

	return &server{
		cfg: config{
			apiKey:  "test-key",
			baseURL: up.URL,
			model:   "deepseek-chat",
			addr:    ":0",
		},
		client: &http.Client{Timeout: 5 * time.Second},
	}
}

func TestChatSuccess(t *testing.T) {
	s := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("неожиданный путь: %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q", got)
		}

		var payload struct {
			Model    string        `json:"model"`
			Messages []chatMessage `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if payload.Model != "deepseek-chat" {
			t.Errorf("model = %q", payload.Model)
		}
		if len(payload.Messages) != 1 || payload.Messages[0].Content != "привет" {
			t.Errorf("messages = %+v", payload.Messages)
		}

		_, _ = w.Write([]byte(`{"model":"deepseek-chat","choices":[{"message":{"role":"assistant","content":"Привет!"}}]}`))
	})

	body := strings.NewReader(`{"messages":[{"role":"user","content":"привет"}]}`)
	rec := httptest.NewRecorder()
	s.chat(rec, httptest.NewRequest(http.MethodPost, "/api/chat", body))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}

	var got chatResponse
	if err := json.NewDecoder(rec.Body).Decode(&got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got.Reply != "Привет!" {
		t.Errorf("reply = %q", got.Reply)
	}
}

func TestChatValidation(t *testing.T) {
	s := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		t.Error("upstream не должен вызываться")
	})

	cases := []string{
		`{}`,
		`{"messages":[]}`,
		`{"messages":[{"role":"hacker","content":"x"}]}`,
		`{"messages":[{"role":"user","content":"   "}]}`,
	}

	for _, c := range cases {
		rec := httptest.NewRecorder()
		s.chat(rec, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(c)))
		if rec.Code != http.StatusBadRequest {
			t.Errorf("body %s: status = %d, ожидался 400", c, rec.Code)
		}
	}
}

func TestChatUpstreamError(t *testing.T) {
	s := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"invalid api key"}}`))
	})

	body := strings.NewReader(`{"messages":[{"role":"user","content":"привет"}]}`)
	rec := httptest.NewRecorder()
	s.chat(rec, httptest.NewRequest(http.MethodPost, "/api/chat", body))

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "invalid api key") {
		t.Errorf("body = %s", rec.Body.String())
	}
}

func TestLoadAPIKeyFromFile(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "")

	path := filepath.Join(t.TempDir(), "api_key.txt")
	if err := os.WriteFile(path, []byte("\ufeff  sk-file-key\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("DEEPSEEK_API_KEY_FILE", path)

	key, err := loadAPIKey()
	if err != nil {
		t.Fatalf("loadAPIKey: %v", err)
	}
	if key != "sk-file-key" {
		t.Errorf("key = %q", key)
	}
}

func TestLoadAPIKeyEnvWins(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "sk-from-env")
	t.Setenv("DEEPSEEK_API_KEY_FILE", filepath.Join(t.TempDir(), "missing.txt"))

	key, err := loadAPIKey()
	if err != nil {
		t.Fatalf("loadAPIKey: %v", err)
	}
	if key != "sk-from-env" {
		t.Errorf("key = %q", key)
	}
}

func TestLoadAPIKeyMissingFile(t *testing.T) {
	t.Setenv("DEEPSEEK_API_KEY", "")
	t.Setenv("DEEPSEEK_API_KEY_FILE", filepath.Join(t.TempDir(), "missing.txt"))

	if _, err := loadAPIKey(); err == nil {
		t.Fatal("ожидалась ошибка для отсутствующего файла")
	}
}

func TestLoadDotEnv(t *testing.T) {
	os.Unsetenv("AI_ADVENT_TEST_KEY")
	os.Unsetenv("AI_ADVENT_TEST_OTHER")
	t.Cleanup(func() {
		os.Unsetenv("AI_ADVENT_TEST_KEY")
		os.Unsetenv("AI_ADVENT_TEST_OTHER")
	})

	dir := t.TempDir()
	path := dir + "/.env"
	content := "\ufeff# comment\nexport AI_ADVENT_TEST_KEY=\"sk-123\"\nAI_ADVENT_TEST_OTHER='value'\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	if err := loadDotEnv(path); err != nil {
		t.Fatalf("loadDotEnv: %v", err)
	}
	if got := os.Getenv("AI_ADVENT_TEST_KEY"); got != "sk-123" {
		t.Errorf("AI_ADVENT_TEST_KEY = %q", got)
	}
	if got := os.Getenv("AI_ADVENT_TEST_OTHER"); got != "value" {
		t.Errorf("AI_ADVENT_TEST_OTHER = %q", got)
	}
}
