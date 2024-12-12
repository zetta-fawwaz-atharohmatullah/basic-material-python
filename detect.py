import dl_translate as dlt
import nltk
from langdetect import detect
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LanguageTranslator:
    
    def __init__(self, model_name="m2m100", num_threads=4):
        """
        Initialize the dl-translate TranslationModel and set up multi-threading.
        Args:
            model_name (str): The model to use for translation (default is 'm2m100').
            num_threads (int): Number of threads for parallel processing.
        """
        try:
            print("Loading the translation model. This may take a while on the first run...")
            self.translator = dlt.TranslationModel(model_name)
            self.num_threads = num_threads
            print("Model loaded successfully!")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            self.translator = None

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text using langdetect.
        Args:
            text (str): The text for which to detect the language.
        Returns:
            str: Detected ISO language code (like 'en', 'es', 'fr', etc.).
        """
        try:
            logging.info("Start detect the language")
            detected_lang_code = detect(text)
            logging.info("Finished detect the language")
            return detected_lang_code
        except Exception as e:
            #print(f"Error detecting language for text: {text}\nError: {e}")
            logging.exception(f"Error detecting language for text: {text}\nError: {e}")
            return "unknown"

    def translate_text(self, text: str, target_lang: str = dlt.lang.ENGLISH) -> str:
        """
        Detects the language of the text and translates it into the target language.
        Args:
            text (str): The text to detect and translate.
            target_lang (str): The target language for translation (default is English).
        Returns:
            str: The translated text.
        """
        try:
            logging.info("Start translate the language")
            detected_lang_code = self.detect_language(text)
            if detected_lang_code == "unknown":
                raise ValueError(f"Unable to detect source language for text: {text}")

            # tokenize for better processing
            sents = nltk.tokenize.sent_tokenize(text, "english")
            translated_text = self._translate_parallel(sents, source=detected_lang_code, target=target_lang)
            logging.info("Finished translate the language")
            return translated_text
        except Exception as e:
            print(f"Error translating text: {text}\nError: {e}")
            logging.exception(f"Error translating text: {text}\nError: {e}")
            return "translation_failed"

    def _translate_parallel(self, sents: list[str], source: str, target: str) -> str:
        """
        Translates multiple sentences in parallel using ThreadPoolExecutor.
        Args:
            sents (list[str]): List of sentences to be translated.
            source (str): Source language.
            target (str): Target language.
        Returns:
            str: Translated text combined from all sentences.
        """
        logging.info("Start helper function translate parallel")
        results = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # submit each sentence for translation in a separate thread
            futures = [executor.submit(self.translator.translate, sent, source=source, target=target) for sent in sents]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error during translation: {e}")
        logging.info("Finished helper function translate parallel")
        return " ".join(results)

    def batch_translate(self, texts: list[str], target_lang: str = dlt.lang.ENGLISH) -> list[dict]:
        """
        Translates a batch of texts.
        Args:
            texts (list[str]): List of texts to be translated.
            target_lang (str): The target language for translation (default is English).
        Returns:
            list[dict]: List of dictionaries with original, detected, and translated texts.
        """
        logging.info("Start batch translate")
        results = []
        for text in texts:
            translated_text = self.translate_text(text, target_lang)
            results.append({
                'original_text': text,
                'translated_text': translated_text
            })
        logging.exception("Finished batch translate")
        return results


if __name__ == "__main__":
    # Instantiate the language translator with 4 threads
    translator = LanguageTranslator(num_threads=4)

    # Sample texts 
    test_texts = [
       "Hola, ¿cómo estás? Espero que te encuentres muy bien el día de hoy. Este es un ejemplo de texto en español que sirve para ilustrar la construcción de oraciones más largas y detalladas. La intención es proporcionar una muestra clara y completa que permita observar el uso adecuado de la gramática, la puntuación y la coherencia en la redacción. Este tipo de texto puede ser útil para aprender, practicar o mejorar el dominio del idioma español.",
       "Bonjour, comment ça va ? Ce texte est écrit en français pour illustrer l'usage de la langue dans une conversation quotidienne. La salutation 'Bonjour' est utilisée pour dire 'Hello' ou 'Good morning' en anglais, tandis que 'comment ça va ?' signifie 'how are you?' ou 'how's it going?'. Ce type d'expression est couramment utilisé pour engager une discussion informelle et amicale. Le fait de préciser que 'Ce texte est écrit en français' met en évidence l'intention de signaler la langue utilisée, ce qui peut être utile dans un contexte éducatif ou de traduction.",
    ]

    print("\n--- Batch Translation ---\n")
    results = translator.batch_translate(test_texts, target_lang=dlt.lang.ENGLISH)

    for i, result in enumerate(results):
        print(f"Text #{i+1}")
        detect_lang = translator.detect_language(result['original_text'])
        print(f"Detect Language: {detect_lang}")
        print(f"Original: {result['original_text']}\n")
        print(f"Translated: {result['translated_text']}\n")
    logging.info("finished main script")
