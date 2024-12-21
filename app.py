import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")         # opsiyonel
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID") # opsiyonel

client = OpenAI(
    api_key=OPENAI_API_KEY,
    organization=OPENAI_ORG_ID,
    project=OPENAI_PROJECT_ID
)

app = Flask(__name__)
CORS(app)
@app.route('/detect-language', methods=['POST'])
def detect_language():
    """
    Detects the language of the given text.
    """
    try:
        data = request.get_json()
        text_to_detect = data.get("text_to_detect", "")

        if not text_to_detect.strip():
            return jsonify({"error": "No text provided."}), 400

        # Dil algılama isteği
        system_message = (
            "You are a language detection tool."
            "\nYour job is to determine the primary language of the provided text."
            "\nRespond with only the detected language as a single word, such as 'Chinese' or 'English'."
        )

        user_message = f"What is the language of this text? {text_to_detect}"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0
        )

        # Tespit edilen dili sadece bir kelime olarak al
        detected_language = response.choices[0].message.content.strip()

        return jsonify({"detected_language": detected_language}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/translate', methods=['POST'])
def translate_text():
    """
    Translates a given text (with automatic source language detection)
    into the specified target language.
    Prioritizes context, fluency, and detail retention.
    Adds parenthetical explanations for culturally or linguistically unique terms.
    """
    try:
        data = request.get_json()
        text_to_translate = data.get("text_to_translate", "")
        target_language = data.get("target_language", "en")

        # System message with rules and tone
        system_message = (
    "You are an advanced translator. Your job is to:"
    "\n1. Automatically detect the source language of the provided text."
    "\n2. Translate it into the specified target language with excellent grammar, fluency, and coherence."
    "\n3. Preserve the original meaning, tone, and context, but adapt to the target language’s natural style."
    "\n4. Adapt the translation to the relevant domain (e.g., finance, politics, technology) and use appropriate terminology. For culturally unique or technical terms, provide a brief parenthetical explanation if necessary."
    "\n5. Keep proper nouns (e.g., names, links like x.com/..., symbols) as-is."
    "\n6. Ensure numerical or financial data remains accurate."
    "\n7. Clearly reflect who is performing any actions mentioned in the text, while maintaining the natural fluency and coherence of the target language."
    "\n8. Produce a polished, professional, and culturally appropriate translation without repeating instructions or adding extraneous comments."
            )


        # User input message
        user_message = (
    f"{text_to_translate}\n"
    f"Target language: {target_language}\n"
    "Please translate this text into the target language. Provide the closest natural equivalent, and if culturally significant, include a brief parenthetical explanation."
     "Please translate this text into the target language using relevant domain-specific terminology."
            )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        response = client.chat.completions.create(
            model="gpt-4o",  # veya gpt-4o-mini
            messages=messages,
            temperature=0.1  # Daha deterministik, tutarlı sonuç
        )

        translated_text = response.choices[0].message.content.strip()
        return jsonify({"translated_text": translated_text}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Heroku'nun sağladığı $PORT ortam değişkenini alın
    port = int(os.environ.get("PORT", 5000))
    # Flask uygulamasını bu portta başlatın
    app.run(host="0.0.0.0", port=port, debug=True)
