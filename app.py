import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
import base_prompts

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")         # Opsiyonel
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID") # Opsiyonel

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
    Gönderilen metnin dilini otomatik olarak tespit eder.
    """
    try:
        data = request.get_json()
        text_to_detect = data.get("text_to_detect", "")

        if not text_to_detect.strip():
            return jsonify({"error": "No text provided."}), 400

        # GPT'ye dil tespiti için gerekli sistem komutu
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

        detected_language = response.choices[0].message.content.strip()

        return jsonify({"detected_language": detected_language}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/translate', methods=['POST'])
def translate_text():
    """
    Gelen metni, isteğe bağlı olarak kullanıcıdan alınan kaynak dil bilgisine
    veya otomatik dil tespitine göre hedef dile çevirir.
    Çeviri, metnin orijinal bağlamını koruyarak kültürel/teknik açıklamalar yapar.
    """
    try:
        data = request.get_json()
        text_to_translate = data.get("text_to_translate", "")
        target_language = data.get("target_language", "en")
        source_language = data.get("source_language", "")  # Kullanıcıdan manuel kaynak dil

        if not text_to_translate.strip():
            return jsonify({"error": "No text to translate provided."}), 400

        # Eğer kullanıcı manuel kaynak dil belirtmezse, otomatik olarak tespit et
        if not source_language.strip():
            # Tespit için GPT'ye yine bir istek yapıyoruz (detect_language fonk. kullanabilir veya benzer şekilde yapabilirsiniz)
            detect_system_message = (
                "You are a language detection tool."
                "\nYour job is to determine the primary language of the provided text."
                "\nRespond with only the detected language as a single word, such as 'Chinese' or 'English'."
            )
            detect_user_message = f"What is the language of this text? {text_to_translate}"

            detect_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": detect_system_message},
                    {"role": "user", "content": detect_user_message},
                ],
                temperature=0
            )

            source_language = detect_response.choices[0].message.content.strip()

        # Artık source_language elimizde: Kullanıcı verdi ya da otomatik tespit ettik
        # Şimdi çeviri isteği için gerekli sistem komutunu hazırlıyoruz
        system_message = base_prompts.base_system_message 

        # Kullanıcı mesajı: Çeviri işlemi için kaynak dili belirtip hedef dilde çeviri istiyor
        user_message = (
            f"Source language: {source_language}\n"
            f"Target language: {target_language}\n"
            f"Text to translate: {text_to_translate}\n"
            "Please produce a natural and context-appropriate translation using the points above."
        )

        # GPT'ye çeviri isteğini gönderiyoruz
        translation_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1  # Daha deterministik, tutarlı sonuç
        )

        translated_text = translation_response.choices[0].message.content.strip()

        return jsonify({
            "detected_source_language": source_language,
            "translated_text": translated_text
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Heroku'nun atadığı PORT'u al, yoksa 5000 kullan
    app.run(host="0.0.0.0", port=port)

