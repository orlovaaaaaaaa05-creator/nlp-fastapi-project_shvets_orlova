import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.corpus import wordnet

# Скачиваем необходимые ресурсы
try:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    nltk.download('wordnet')
except:
    print("NLTK resources already downloaded or error occurred")

class NLTKProcessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def tokenize(self, text):
        """Токенизация текста"""
        return word_tokenize(text)
    
    def stem(self, text):
        """Стемминг текста"""
        tokens = self.tokenize(text)
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize(self, text):
        """Лемматизация текста"""
        tokens = self.tokenize(text)
        
        # Функция для преобразования тегов
        def get_wordnet_pos(tag):
            if tag.startswith('J'):
                return wordnet.ADJ
            elif tag.startswith('V'):
                return wordnet.VERB
            elif tag.startswith('N'):
                return wordnet.NOUN
            elif tag.startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN
        
        # Получаем POS теги
        pos_tags = pos_tag(tokens)
        
        # Лемматизируем с учетом POS
        lemmas = []
        for token, tag in pos_tags:
            pos = get_wordnet_pos(tag)
            lemma = self.lemmatizer.lemmatize(token, pos=pos)
            lemmas.append(lemma)
        
        return lemmas
    
    def pos_tag(self, text):
        """POS тегинг"""
        tokens = self.tokenize(text)
        return pos_tag(tokens)
    
    def ner(self, text):
        """Распознавание именованных сущностей"""
        tokens = self.tokenize(text)
        pos_tags = pos_tag(tokens)
        
        # Используем ne_chunk для NER
        tree = ne_chunk(pos_tags)
        
        # Извлекаем именованные сущности
        entities = []
        current_entity = []
        entity_type = None
        
        for node in tree:
            if hasattr(node, 'label'):
                if current_entity:
                    entities.append({
                        'entity': ' '.join(current_entity),
                        'type': entity_type
                    })
                    current_entity = []
                
                entity_type = node.label()
                current_entity.append(node[0][0])
            else:
                if current_entity:
                    current_entity.append(node[0])
                else:
                    if entity_type and current_entity:
                        entities.append({
                            'entity': ' '.join(current_entity),
                            'type': entity_type
                        })
                    current_entity = []
        
        if current_entity and entity_type:
            entities.append({
                'entity': ' '.join(current_entity),
                'type': entity_type
            })
        
        return entities