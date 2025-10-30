import requests
from openai import OpenAI

def main(user_prompt, gpt_key):
    """
    Agent that calls the Japanese translation Lambda and explains the translation.
    """
    lambda_url = "https://jlz3boo2wgdawr3puaqsfx4zha0lcuib.lambda-url.eu-north-1.on.aws/"
    
    headers = {"Content-Type": "application/json"}

    try:
        # Step 1: Call translation Lambda
        payload = {"user_prompt": user_prompt, "gpt_key": gpt_key}
        response = requests.post(lambda_url, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        lambda_data = response.json()

        translation_full = lambda_data.get("message", "").strip()
        japanese_text = lambda_data.get("data", {}).get("japanese", "")
        romaji_text = lambda_data.get("data", {}).get("romaji", "")

        if not japanese_text:
            return {"success": False, "message": "No Japanese translation returned from Lambda."}

        # Step 2: Ask GPT to explain the Japanese translation (ignore Romaji)
        client = OpenAI(api_key=gpt_key)
        explain_prompt = f"""
You are a Japanese linguist. Analyze only the Japanese translation (ignore the Romaji).

Original English:
{user_prompt}

Japanese Translation (Native Script):
{japanese_text}

Explain in English:
- Accuracy and naturalness of translation
- Tone and formality
- Any cultural nuances or interesting word choices
        """

        explanation_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You explain Japanese translations clearly and concisely in English."},
                {"role": "user", "content": explain_prompt.strip()},
            ],
            temperature=0.4,
        )

        explanation = explanation_response.choices[0].message.content.strip()

        return {
            "success": True,
            "message": f"{explanation}",
            "data": {
                "original": user_prompt,
                "japanese": japanese_text,
                "romaji": romaji_text,
                "explanation": explanation,
            },
        }

    except Exception as e:
        return {"success": False, "message": str(e), "data": {}}

