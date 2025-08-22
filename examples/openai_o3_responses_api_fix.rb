require "dotenv/load"
require "langchain"

# This example demonstrates the fix for using o3 model with Responses API
# The o3 model (and other reasoning models) are not supported by the Responses API

puts "OpenAI o3 Model with Responses API Fix Demo"
puts "=" * 50

# This will now raise an error during initialization
puts "\n1. Trying to initialize with o3 model and Responses API (should fail)"
puts "-" * 60

begin
  llm = Langchain::LLM::OpenAI.new(
    api_key: ENV["OPENAI_API_KEY"],
    use_responses_api: true,
    default_options: {chat_model: "o3"}
  )
  puts "❌ This should have failed!"
rescue ArgumentError => e
  puts "✅ Correctly caught error: #{e.message}"
end

# This will also raise an error when trying to use o3 model in chat
puts "\n2. Trying to use o3 model in chat with Responses API (should fail)"
puts "-" * 60

begin
  llm = Langchain::LLM::OpenAI.new(
    api_key: ENV["OPENAI_API_KEY"],
    use_responses_api: true
  )
  
  response = llm.chat(
    messages: [{role: "user", content: "Hello"}],
    model: "o3"
  )
  puts "❌ This should have failed!"
rescue ArgumentError => e
  puts "✅ Correctly caught error: #{e.message}"
end

# This works fine - o3 model with traditional API
puts "\n3. Using o3 model with traditional Chat Completions API (should work)"
puts "-" * 60

begin
  llm = Langchain::LLM::OpenAI.new(
    api_key: ENV["OPENAI_API_KEY"],
    use_responses_api: false,
    default_options: {chat_model: "o3"}
  )
  puts "✅ Successfully initialized with o3 model and traditional API"
  
  # Note: This would make an actual API call if you have a valid API key
  # For demo purposes, we'll just show the initialization works
  puts "   The LLM is ready to use with o3 model"
rescue => e
  puts "❌ Unexpected error: #{e.message}"
end

# This also works fine - regular model with Responses API
puts "\n4. Using regular model (gpt-4o-mini) with Responses API (should work)"
puts "-" * 60

begin
  llm = Langchain::LLM::OpenAI.new(
    api_key: ENV["OPENAI_API_KEY"],
    use_responses_api: true,
    default_options: {chat_model: "gpt-4o-mini"}
  )
  puts "✅ Successfully initialized with gpt-4o-mini model and Responses API"
  
  # Note: This would make an actual API call if you have a valid API key
  # For demo purposes, we'll just show the initialization works
  puts "   The LLM is ready to use with Responses API"
rescue => e
  puts "❌ Unexpected error: #{e.message}"
end

puts "\n" + "=" * 50
puts "Summary:"
puts "- Reasoning models (o1, o3) are NOT supported by the Responses API"
puts "- Use traditional Chat Completions API for reasoning models"
puts "- Use Responses API for regular models (gpt-4o-mini, gpt-4o, etc.)"
puts "- The fix now provides clear error messages to guide users" 