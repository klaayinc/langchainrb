# frozen_string_literal: true

module Langchain::LLM
  class OpenAIResponsesResponse < BaseResponse
    def model
      raw_response["model"]
    end

    def created_at
      if raw_response.dig("created")
        Time.at(raw_response.dig("created"))
      end
    end

    def completion
      # Responses API uses 'output' instead of 'choices'
      output = raw_response.dig("output")
      return nil unless output&.any?
      
      # Look for message type output first
      message_output = output.find { |item| item["type"] == "message" }
      if message_output && message_output.dig("content")
        # Handle array of content items
        if message_output["content"].is_a?(Array)
          text_content = message_output["content"].find { |c| c["type"] == "output_text" }
          return text_content["text"] if text_content
        end
        
        # Handle direct content string (fallback)
        return message_output["content"] if message_output["content"].is_a?(String)
      end
      
      nil
    end

    def role
      # Responses API doesn't have explicit roles in the same way
      "assistant"
    end

    def chat_completion
      completion
    end

    def tool_calls
      # Responses API uses 'output' with 'function_call' type
      output = raw_response.dig("output")
      return [] unless output&.any?
      
      function_calls = output.select { |item| item["type"] == "function_call" }
      function_calls.map do |fc|
        {
          "id" => fc["call_id"],
          "type" => "function",
          "function" => {
            "name" => fc["name"],
            "arguments" => fc["arguments"] || "{}"
          }
        }
      end
    end

    def embedding
      embeddings&.first
    end

    def completions
      # Responses API uses 'output' instead of 'choices'
      # Transform to Chat Completions format for compatibility
      output = raw_response.dig("output")
      return [] unless output&.any?
      
      [
        {
          "message" => {
            "role" => "assistant",
            "content" => completion,
            "tool_calls" => tool_calls.any? ? tool_calls : nil
          }.compact,
          "finish_reason" => determine_finish_reason,
          "index" => 0
        }
      ]
    end

    def chat_completions
      completions
    end

    def embeddings
      raw_response.dig("data")&.map { |datum| datum.dig("embedding") }
    end

    def prompt_tokens
      raw_response.dig("usage", "input_tokens") || raw_response.dig("usage", "prompt_tokens")
    end

    def completion_tokens
      raw_response.dig("usage", "output_tokens") || raw_response.dig("usage", "completion_tokens")
    end

    def total_tokens
      raw_response.dig("usage", "total_tokens")
    end

    private

    def determine_finish_reason
      # Determine finish reason based on response status and output types
      return "stop" if raw_response.dig("status") == "completed"
      return "tool_calls" if tool_calls.any?
      "stop"
    end
  end
end 