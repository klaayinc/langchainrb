# frozen_string_literal: true

module Langchain::LLM
  # LLM interface for OpenAI APIs: https://platform.openai.com/overview
  #
  # Gem requirements:
  #    gem "ruby-openai", "~> 6.3.0"
  #
  # Usage:
  #    llm = Langchain::LLM::OpenAI.new(
  #      api_key: ENV["OPENAI_API_KEY"],
  #      llm_options: {}, # Available options: https://github.com/alexrudall/ruby-openai/blob/main/lib/openai/client.rb#L5-L13
  #      default_options: {}
  #    )
  class OpenAI < Base
    DEFAULTS = {
      n: 1,
      chat_model: "gpt-4o-mini",
      embedding_model: "text-embedding-3-small"
    }.freeze

    EMBEDDING_SIZES = {
      "text-embedding-ada-002" => 1536,
      "text-embedding-3-large" => 3072,
      "text-embedding-3-small" => 1536
    }.freeze

    # Initialize an OpenAI LLM instance
    #
    # @param api_key [String] The API key to use
    # @param client_options [Hash] Options to pass to the OpenAI::Client constructor
    def initialize(api_key:, llm_options: {}, default_options: {})
      depends_on "ruby-openai", req: "openai"

      llm_options[:log_errors] = Langchain.logger.debug? unless llm_options.key?(:log_errors)

      @client = ::OpenAI::Client.new(access_token: api_key, **llm_options) do |f|
        f.response :logger, Langchain.logger, {headers: true, bodies: true, errors: true}
      end

      @defaults = DEFAULTS.merge(default_options)
      chat_parameters.update(
        model: {default: @defaults[:chat_model]},
        logprobs: {},
        top_logprobs: {},
        n: {default: @defaults[:n]},
        temperature: {default: @defaults[:temperature]},
        user: {},
        response_format: {default: @defaults[:response_format]},
        reasoning_effort: {default: @defaults[:reasoning_effort]}
      )
      chat_parameters.ignore(:top_k)
    end

    # Generate an embedding for a given text
    #
    # @param text [String] The text to generate an embedding for
    # @param model [String] ID of the model to use
    # @param encoding_format [String] The format to return the embeddings in. Can be either float or base64.
    # @param user [String] A unique identifier representing your end-user
    # @return [Langchain::LLM::OpenAIResponse] Response object
    def embed(
      text:,
      model: defaults[:embedding_model],
      encoding_format: nil,
      user: nil,
      dimensions: @defaults[:dimensions]
    )
      raise ArgumentError.new("text argument is required") if text.empty?
      raise ArgumentError.new("model argument is required") if model.empty?
      raise ArgumentError.new("encoding_format must be either float or base64") if encoding_format && %w[float base64].include?(encoding_format)

      parameters = {
        input: text,
        model: model
      }
      parameters[:encoding_format] = encoding_format if encoding_format
      parameters[:user] = user if user

      if dimensions
        parameters[:dimensions] = dimensions
      elsif EMBEDDING_SIZES.key?(model)
        parameters[:dimensions] = EMBEDDING_SIZES[model]
      end

      # dimensions parameter not supported by text-embedding-ada-002 model
      parameters.delete(:dimensions) if model == "text-embedding-ada-002"

      response = with_api_error_handling do
        client.embeddings(parameters: parameters)
      end

      Langchain::LLM::OpenAIResponse.new(response)
    end

    # rubocop:disable Style/ArgumentsForwarding
    # Generate a completion for a given prompt
    #
    # @param prompt [String] The prompt to generate a completion for
    # @param params [Hash] The parameters to pass to the `chat()` method
    # @return [Langchain::LLM::OpenAIResponse] Response object
    # @deprecated Use {chat} instead.
    def complete(prompt:, **params)
      Langchain.logger.warn "DEPRECATED: `Langchain::LLM::OpenAI#complete` is deprecated, and will be removed in the next major version. Use `Langchain::LLM::OpenAI#chat` instead."

      if params[:stop_sequences]
        params[:stop] = params.delete(:stop_sequences)
      end
      # Should we still accept the `messages: []` parameter here?
      messages = [{role: "user", content: prompt}]
      chat(messages: messages, **params)
    end

    # rubocop:enable Style/ArgumentsForwarding

    # Generate a chat completion for given messages.
    #
    # @param [Hash] params unified chat parmeters from [Langchain::LLM::Parameters::Chat::SCHEMA]
    # @option params [Array<Hash>] :messages List of messages comprising the conversation so far
    # @option params [String] :model ID of the model to use
    def chat(params = {}, &block)
      parameters = chat_parameters.to_params(params)
      parameters[:metadata] = params[:metadata] if params[:metadata]

      raise ArgumentError.new("messages argument is required") if Array(parameters[:messages]).empty?
      raise ArgumentError.new("model argument is required") if parameters[:model].to_s.empty?
      if parameters[:tool_choice] && Array(parameters[:tools]).empty?
        raise ArgumentError.new("'tool_choice' is only allowed when 'tools' are specified.")
      end

      # Enforce OpenAI reasoning model constraints (e.g., o1/o3):
      #  - temperature must be 1
      #  - parallel_tool_calls is unsupported and must be omitted entirely
      if reasoning_model_name?(parameters[:model])
        if parameters[:temperature] && parameters[:temperature] != 1
          Langchain.logger.info "Forcing temperature=1 for OpenAI reasoning model #{parameters[:model]} (was #{parameters[:temperature]})"
        end
        parameters[:temperature] = 1

        if parameters.key?(:parallel_tool_calls)
          Langchain.logger.info "Removing unsupported 'parallel_tool_calls' for OpenAI reasoning model #{parameters[:model]}"
          parameters.delete(:parallel_tool_calls)
        end
      end

      if block
        @response_chunks = []
        parameters[:stream_options] = {include_usage: true}
        parameters[:stream] = proc do |chunk, _bytesize|
          chunk_content = chunk.dig("choices", 0) || {}
          @response_chunks << chunk
          yield chunk_content
        end
      end

      response = with_api_error_handling do
        client.chat(parameters: parameters)
      end

      response = response_from_chunks if block
      reset_response_chunks

      Langchain::LLM::OpenAIResponse.new(response)
    end

    # Generate a summary for a given text
    #
    # @param text [String] The text to generate a summary for
    # @return [String] The summary
    def summarize(text:)
      prompt_template = Langchain::Prompt.load_from_path(
        file_path: Langchain.root.join("langchain/llm/prompts/summarize_template.yaml")
      )
      prompt = prompt_template.format(text: text)

      complete(prompt: prompt)
    end

    def default_dimensions
      @defaults[:dimensions] || EMBEDDING_SIZES.fetch(defaults[:embedding_model])
    end

    private

    attr_reader :response_chunks

    def reset_response_chunks
      @response_chunks = []
    end

    def with_api_error_handling
      attempts = Integer(@defaults[:retry_attempts] || 0)
      base_backoff = Float(@defaults[:retry_backoff_base] || 0.5)
      tries = 0

      begin
        response = yield
        return if response.nil? || response.empty?

        raise Langchain::LLM::ApiError.new "OpenAI API error: #{response.dig("error", "message")}" if response&.dig("error")

        response
      rescue Faraday::Error => e
        if Object.const_defined?("Harpoon")
          Harpoon.create(issue: "openai_transient_error", event: e.class, payload: (e.respond_to?(:response) ? e.response : nil))
        end
        # Retry on transient HTTP statuses/timeouts/connection failures if configured
        res_status = (e.response && e.response[:status])
        transient_status = res_status && (res_status == 429 || (res_status >= 500 && res_status < 600))
        is_timeout = defined?(Faraday::TimeoutError) && e.is_a?(Faraday::TimeoutError)
        is_connection_failed = defined?(Faraday::ConnectionFailed) && e.is_a?(Faraday::ConnectionFailed)

        if tries < attempts && (transient_status || is_timeout || is_connection_failed)
          sleep(base_backoff * (2 ** tries))
          tries += 1
          retry
        end

        # Timeouts/connection failures without response or after exhausting retries
        if is_timeout
          new_e = Langchain::LLM::TimeoutError.new("OpenAI API timeout: #{e.message}")
          Sentry.capture_exception(new_e) if Object.const_defined?("Sentry")
          raise new_e
        elsif is_connection_failed
          new_e = Langchain::LLM::ConnectionError.new("OpenAI API connection failed: #{e.message}")
          Sentry.capture_exception(new_e) if Object.const_defined?("Sentry")
          raise new_e
        end

        raise unless e.response.respond_to?(:dig)

        req_method = e.response.dig(:request, :method)
        req_url = e.response.dig(:request, :url)
        req_params = e.response.dig(:request, :params)
        req_headers = e.response.dig(:request, :headers)
        req_body = e.response.dig(:request, :body)
        res_headers = e.response[:headers]
        res_body = e.response[:body]

        message = <<~ERR
          OpenAI API error: Server responded with #{res_status}
          --- Request ---
          #{req_method&.upcase} #{req_url} #{req_params}
          Headers:
          #{req_headers}
          Body:
          #{req_body}
          --- Response ---
          Headers:
          #{res_headers}
          Body:
          #{res_body}
        ERR

        error_class = case res_status
        when 400 then Langchain::LLM::BadRequestError
        when 401 then Langchain::LLM::UnauthorizedError
        when 403 then Langchain::LLM::ForbiddenError
        when 404 then Langchain::LLM::NotFoundError
        when 409 then Langchain::LLM::ConflictError
        when 422 then Langchain::LLM::UnprocessableEntityError
        when 429 then Langchain::LLM::RateLimitError
        when 503 then Langchain::LLM::ServiceUnavailableError
        when 500..599 then Langchain::LLM::ServerError
        else Langchain::LLM::ApiError
        end

        new_e = error_class.new(message)

        if Object.const_defined?("Sentry")
          Sentry.capture_exception(new_e)
        end

        raise new_e
      end
    end

    def response_from_chunks
      grouped_chunks = @response_chunks
        .group_by { |chunk| chunk.dig("choices", 0, "index") }
        .except(nil) # the last chunk (that contains the token usage) has no index
      final_choices = grouped_chunks.map do |index, chunks|
        {
          "index" => index,
          "message" => {
            "role" => "assistant",
            "content" => chunks.map { |chunk| chunk.dig("choices", 0, "delta", "content") }.join,
            "tool_calls" => tool_calls_from_choice_chunks(chunks)
          }.compact,
          "finish_reason" => chunks.last.dig("choices", 0, "finish_reason")
        }
      end
      @response_chunks.first&.slice("id", "object", "created", "model")&.merge({"choices" => final_choices, "usage" => @response_chunks.last["usage"]})
    end

    def tool_calls_from_choice_chunks(choice_chunks)
      tool_call_chunks = choice_chunks.select { |chunk| chunk.dig("choices", 0, "delta", "tool_calls") }
      return nil if tool_call_chunks.empty?

      tool_call_chunks.group_by { |chunk| chunk.dig("choices", 0, "delta", "tool_calls", 0, "index") }.map do |index, chunks|
        first_chunk = chunks.first

        {
          "id" => first_chunk.dig("choices", 0, "delta", "tool_calls", 0, "id"),
          "type" => first_chunk.dig("choices", 0, "delta", "tool_calls", 0, "type"),
          "function" => {
            "name" => first_chunk.dig("choices", 0, "delta", "tool_calls", 0, "function", "name"),
            "arguments" => chunks.map { |chunk| chunk.dig("choices", 0, "delta", "tool_calls", 0, "function", "arguments") }.join
          }
        }
      end
    end

    def reasoning_model_name?(model_name)
      model_name.to_s.start_with?("o1", "o3")
    end
  end
end
