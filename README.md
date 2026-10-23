# AnonymAIzer

Applicazione Streamlit locale per anonimizzare testi con **Rizzo-PII 0.3B**.

## Avvio su Windows / Anaconda

```bat
conda activate envdata
cd /d "G:\Il mio Drive\TOOLS\anonymAIzer"
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

In alternativa, apri il terminale nella cartella dove hai estratto lo ZIP ed esegui gli ultimi due comandi.

### Percorso modello

Per default l'app cerca il modello in:

`C:\AI\models\rizzo-pii-0.3B`

Il percorso si può cambiare tramite variabile d'ambiente `ANONYM_AIZER_MODEL`.
La cartella deve contenere `config.json`, file tokenizer e `model.safetensors`.

## Funzioni

- Incolla testo e anonimizza mediante la pipeline di token classification locale;
- tutte le categorie sono selezionate all’avvio, comprese organizzazioni e città; puoi deselezionare quelle da lasciare visibili;
- segnaposto coerenti tra ripetizioni dello stesso valore;
- opzionalmente ripristina i valori tramite un dizionario mantenuto nella sessione Streamlit;
- copia il testo anonimizzato negli appunti con l’icona del riquadro (poi Ctrl+V), visualizza categorie rilevate e scarica un file TXT;
- semplici controlli regex per email, URL, IPv4, CF e IBAN.

## Limiti e privacy

- Questa è un'integrazione autonoma di `transformers`: **non** include tutta la rete regex/checksum, la redazione PDF e le altre protezioni offerte dall'applicazione ufficiale Rizzo-PII.
- Non vengono implementate verifiche formali dei checksum di IBAN/CF: le espressioni regolari complementari sono solo controlli di formato.
- Il punteggio del modello **non è** una probabilità calibrata che un dato sia PII.
- Una sessione Streamlit mantiene il testo originale nel widget, anche quando il ripristino è disattivato. I dati rimangono inoltre in memoria durante l'elaborazione. Non usare con documenti sensibili senza una valutazione di sicurezza e un controllo umano del risultato.
- Se Streamlit è pubblicato online, i testi vengono elaborati sul **server**, non sul PC del visitatore.
- In questo prototipo non sono previsti upload PDF, autentificazione, logging applicativo né anonimizzazione certificata.

Motore di base: [Rizzo-PII](https://github.com/Rizzo-AI-Academy/rizzo-pii).
