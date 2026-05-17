# notebooks/test_groq.py
# Test de l'API Groq avec Llama 3

import os
from dotenv import load_dotenv
from groq import Groq

# Charger la cle depuis .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERREUR : GROQ_API_KEY non trouvee dans .env")
    exit()

# Creer le client Groq
client = Groq(api_key=api_key)

# Premier appel : question simple
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system",
         "content": "Tu es un assistant medical senegalais. "
                    "Reponds en francais simple. "
                    "Maximum 3 phrases."},
        {"role": "user",
         "content": "Quels sont les symptomes du paludisme ?"}
    ],
    max_tokens=200,
    temperature=0.3
)

print("=== Reponse de Llama 3 ===")
print(response.choices[0].message.content)
print(f"\nTokens utilises : {response.usage.total_tokens}")

# Deuxieme appel : format SenSante
response2 = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system",
         "content": """Tu es un assistant medical senegalais.
Tu recois un diagnostic et des donnees patient.
Explique le resultat en francais simple,
comme un medecin parlerait a son patient.
Sois rassurant mais recommande une consultation.
Maximum 3 phrases.
Ne fais JAMAIS de diagnostic toi-meme.
Tu expliques uniquement le diagnostic fourni."""},
        {"role": "user",
         "content": """Patient : Femme, 28 ans, region Dakar
Symptomes : temperature 39.5, toux, fatigue, maux de tete
Diagnostic du modele : paludisme (probabilite 72%)
Explique ce resultat au patient."""}
    ],
    max_tokens=200,
    temperature=0.3
)

print("\n=== Explication SenSante ===")
print(response2.choices[0].message.content)
# Exercice 1 : Prompt en Wolof
response3 = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system",
         "content": """Tu es un assistant médical sénégalais.
Réponds en wolof simple mélangé de français.
Par exemple : 'Yow, dafa am jigeen bi...' 
Maximum 3 phrases.
Ne fais JAMAIS de diagnostic toi-même."""},
        {"role": "user",
         "content": """Patient : Homme, 35 ans, Ziguinchor
Température : 38.5 C
Diagnostic du modèle : grippe (probabilité 65%)
Explique ce résultat au patient."""}
    ],
    max_tokens=200,
    temperature=0.3
)

print("\n=== Exercice 1 : Réponse en Wolof ===")
print(response3.choices[0].message.content)
# Exercice 2 : Tester différentes températures
for temp in [0.0, 0.5, 1.0]:
    response_temp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system",
             "content": "Tu es un assistant médical sénégalais. Maximum 2 phrases."},
            {"role": "user",
             "content": "Patient : Femme, 28 ans, Dakar. Diagnostic : paludisme (72%). Explique."}
        ],
        max_tokens=150,
        temperature=temp
    )
    print(f"\n=== Exercice 2 : temperature={temp} ===")
    print(response_temp.choices[0].message.content)