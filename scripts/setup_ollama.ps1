# Setup Ollama model for video summarization (Italian, 24GB VRAM)
$ErrorActionPreference = "Stop"

$Model = if ($env:OLLAMA_MODEL) { $env:OLLAMA_MODEL } else { "qwen2.5:32b" }

Write-Host "Pulling Ollama model: $Model"
ollama pull $Model

Write-Host "Running smoke test (Italian)..."
$body = @{
    model = $Model
    messages = @(
        @{ role = "user"; content = "Rispondi in una frase: qual e' il vantaggio di un riassunto gerarchico per video lunghi?" }
    )
    stream = $false
    options = @{
        num_ctx = 4096
        temperature = 0.3
        num_predict = 128
    }
} | ConvertTo-Json -Depth 5

$response = Invoke-RestMethod -Uri "http://localhost:11434/api/chat" -Method Post -Body $body -ContentType "application/json"
Write-Host "Response:" $response.message.content
Write-Host "Ollama setup OK."
