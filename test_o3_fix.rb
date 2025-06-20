#!/usr/bin/env ruby

# Simple test script for the o3 model fix
require_relative 'lib/langchain'

puts "Testing o3 model fix..."

# Test 1: Should fail when using o3 with Responses API during initialization
puts "\nTest 1: Initialization with o3 and Responses API"
begin
  llm = Langchain::LLM::OpenAI.new(
    api_key: "test",
    use_responses_api: true,
    default_options: {chat_model: "o3"}
  )
  puts "❌ Should have failed!"
rescue ArgumentError => e
  puts "✅ Correctly caught: #{e.message}"
end

# Test 2: Should fail when using o3 with Responses API in chat
puts "\nTest 2: Chat with o3 and Responses API"
begin
  llm = Langchain::LLM::OpenAI.new(
    api_key: "test",
    use_responses_api: true
  )
  
  llm.chat(
    messages: [{role: "user", content: "Hello"}],
    model: "o3"
  )
  puts "❌ Should have failed!"
rescue ArgumentError => e
  puts "✅ Correctly caught: #{e.message}"
end

# Test 3: Should work with o3 and traditional API
puts "\nTest 3: o3 with traditional API"
begin
  llm = Langchain::LLM::OpenAI.new(
    api_key: "test",
    use_responses_api: false,
    default_options: {chat_model: "o3"}
  )
  puts "✅ Successfully initialized with o3 and traditional API"
rescue => e
  puts "❌ Unexpected error: #{e.message}"
end

# Test 4: Should work with regular model and Responses API
puts "\nTest 4: Regular model with Responses API"
begin
  llm = Langchain::LLM::OpenAI.new(
    api_key: "test",
    use_responses_api: true,
    default_options: {chat_model: "gpt-4o-mini"}
  )
  puts "✅ Successfully initialized with gpt-4o-mini and Responses API"
rescue => e
  puts "❌ Unexpected error: #{e.message}"
end

puts "\nAll tests completed!" 