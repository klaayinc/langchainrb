require "dotenv/load"
require "langchain"
require "benchmark"

# This example demonstrates the OpenAI Responses API
# The Responses API provides faster response times and better streaming capabilities

puts "OpenAI Responses API Demo"
puts "=" * 50

# Initialize both traditional and Responses API clients
traditional_llm = Langchain::LLM::OpenAI.new(api_key: ENV["OPENAI_API_KEY"])
responses_llm = Langchain::LLM::OpenAI.new(
  api_key: ENV["OPENAI_API_KEY"],
  use_responses_api: true
)

# Test message
messages = [
  {role: "user", content: "Tell me a short joke about programming"}
]

puts "\n1. Basic Chat Completion"
puts "-" * 30

# Traditional API
puts "Traditional API response:"
response = traditional_llm.chat(messages: messages)
puts "Response: #{response.chat_completion}"
puts "Model: #{response.model}"
puts "Tokens used: #{response.total_tokens}"

puts "\nResponses API response:"
response = responses_llm.chat(messages: messages)
puts "Response: #{response.chat_completion}"
puts "Model: #{response.model}"
puts "Tokens used: #{response.total_tokens}"

puts "\n2. Streaming Example"
puts "-" * 30

puts "Traditional API streaming:"
traditional_llm.chat(messages: messages) do |chunk|
  content = chunk.dig("delta", "content")
  print content if content
end
puts

puts "\nResponses API streaming:"
responses_llm.chat(messages: messages) do |chunk|
  content = chunk.dig("delta", "content")
  print content if content
end
puts

puts "\n3. Tool Calling Example"
puts "-" * 30

tools = [
  {
    "type" => "function",
    "function" => {
      "name" => "get_weather",
      "description" => "Get the weather for a location",
      "parameters" => {
        "type" => "object",
        "properties" => {
          "location" => {
            "type" => "string",
            "description" => "The city and state, e.g. San Francisco, CA"
          }
        },
        "required" => ["location"]
      }
    }
  }
]

messages_with_tools = [
  {role: "user", content: "What's the weather like in San Francisco?"}
]

puts "Traditional API with tools:"
response = traditional_llm.chat(messages: messages_with_tools, tools: tools)
if response.tool_calls.any?
  puts "Tool calls: #{response.tool_calls.map { |tc| tc.dig("function", "name") }}"
else
  puts "Response: #{response.chat_completion}"
end

puts "\nResponses API with tools:"
response = responses_llm.chat(messages: messages_with_tools, tools: tools)
if response.tool_calls.any?
  puts "Tool calls: #{response.tool_calls.map { |tc| tc.dig("function", "name") }}"
else
  puts "Response: #{response.chat_completion}"
end

puts "\n4. Performance Comparison"
puts "-" * 30

# Simple performance test
test_messages = [
  {role: "user", content: "Say 'Hello, world!'"}
]

puts "Running performance test (3 iterations each)..."
puts

traditional_times = []
responses_times = []

3.times do |i|
  puts "Iteration #{i + 1}:"
  
  # Traditional API
  traditional_time = Benchmark.realtime do
    traditional_llm.chat(messages: test_messages)
  end
  traditional_times << traditional_time
  puts "  Traditional API: #{traditional_time.round(3)}s"
  
  # Responses API
  responses_time = Benchmark.realtime do
    responses_llm.chat(messages: test_messages)
  end
  responses_times << responses_time
  puts "  Responses API: #{responses_time.round(3)}s"
  puts
end

avg_traditional = traditional_times.sum / traditional_times.size
avg_responses = responses_times.sum / responses_times.size
improvement = ((avg_traditional - avg_responses) / avg_traditional * 100).round(1)

puts "Results:"
puts "  Average Traditional API: #{avg_traditional.round(3)}s"
puts "  Average Responses API: #{avg_responses.round(3)}s"
puts "  Performance improvement: #{improvement}%"

puts "\n" + "=" * 50
puts "The Responses API provides better performance and streaming capabilities"
puts "while maintaining full compatibility with the traditional Chat Completions API."
puts "You can easily switch between them by adding use_responses_api: true" 