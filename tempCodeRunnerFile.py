@app.route('/chat_ai')
def chat_ai():
    # Initialize chat history if not already present in session
    # Each item in history should be {'role': 'user'/'model', 'parts': [text_content]}
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    # You can pass chat_history to the template if you want to pre-load messages,
    # but the current JS adds the initial bot message on DOMContentLoaded.
    return render_template('chat_ai.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"response": "No message received."}), 400

    # Retrieve chat history from session
    # Ensure it's in the format expected by the Gemini API (list of dicts)
    history = session.get('chat_history', [])

    try:
        if not model:
            # If model initialization failed, return an critical error
            return jsonify({"response": "AI model not available on server. Please check server logs for initialization errors."}), 503 # Service Unavailable

        # Start or continue the chat session with the model
        # Using start_chat and passing history is crucial for multi-turn conversations
        chat_session = model.start_chat(history=history)

        # Send the user's message
        response = chat_session.send_message(user_message, safety_settings=SAFETY_SETTINGS)

        # Extract the text response
        ai_response_text = ""
        try:
            ai_response_text = response.text
        except ValueError:
            # Handle cases where response.text might not be directly available
            # e.g., if content was blocked by safety settings or an empty response
            print(f"Gemini response did not contain text content. Raw response: {response}")
            # Check for safety ratings if content was blocked
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                ai_response_text = "Your message was blocked due to safety concerns. Please try rephrasing."
            elif response.candidates and len(response.candidates) > 0 and response.candidates[0].finish_reason == 'SAFETY':
                ai_response_text = "The AI's response was blocked due to safety filters."
            else:
                ai_response_text = "The AI did not provide a complete response."


        # Update session history with both user and AI messages
        # Format: {'role': 'user'/'model', 'parts': [{'text': '...'}]}
        history.append({'role': 'user', 'parts': [{'text': user_message}]})
        history.append({'role': 'model', 'parts': [{'text': ai_response_text}]})
        session['chat_history'] = history # Save updated history back to session

        return jsonify({"response": ai_response_text})

    except genai.types.BlockedPromptException as e:
        print(f"Prompt blocked by safety settings: {e}")
        return jsonify({"response": "Your message was blocked by safety filters. Please try rephrasing."}), 400
    except google.api_core.exceptions.GoogleAPIError as e: # Catch specific Google API errors
        print(f"Gemini API error (GoogleAPIError): {e}")
        return jsonify({"response": f"Error communicating with AI: {e}. Please try again later."}), 500
    except Exception as e: # Catch any other unexpected errors
        print(f"An unexpected error occurred in chat route: {e}")
        return jsonify({"response": f"Sorry, an internal error occurred: {e}. Please try again later."}), 500
