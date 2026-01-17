import requests
import json

class SimpleNLPClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def load_texts_from_file(self, filepath="data/texts.txt"):
        """Загрузка текстов из файла"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            print(f"✅ Загружено {len(texts)} английских текстов")
            return texts
        except FileNotFoundError:
            print(f"❌ Файл {filepath} не найден. Использую пример текстов.")
            return self.get_sample_texts()
        except Exception as e:
            print(f"❌ Ошибка при чтении файла: {e}")
            return self.get_sample_texts()
    
    def get_sample_texts(self):
        """Возвращает пример текстов если файла нет"""
        return [
            "Natural language processing helps computers understand human language.",
            "Machine learning algorithms learn from data.",
            "Deep learning uses neural networks with many layers.",
            "Python is a popular programming language for AI.",
            "FastAPI makes it easy to build web APIs."
        ]
    
    def test_all_endpoints(self):
        """Тестирует все эндпоинты"""
        print("=" * 60)
        print("🚀 ТЕСТИРОВАНИЕ NLP МИКРОСЕРВИСА")
        print("=" * 60)
        
        # Загружаем тексты
        texts = self.load_texts_from_file()
        print(f"📄 Используем {len(texts)} текстов для тестирования\n")
        
        # 1. Тест TF-IDF
        print("1. 📊 Тестируем TF-IDF (на numpy):")
        response = requests.post(f"{self.base_url}/tf-idf", json={
            "texts": texts[:3],  # первые 3 текста
            "max_features": 20
        })
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Успешно! Матрица размером: {data['shape']}")
            if 'vocabulary' in data:
                print(f"   📚 Словарь: {len(data['vocabulary'])} слов")
        else:
            print(f"   ❌ Ошибка {response.status_code}: {response.text[:100]}")
        
        # 2. Тест Bag of Words
        print("\n2. 🛍️ Тестируем Bag of Words (на numpy):")
        response = requests.post(f"{self.base_url}/bag-of-words", json={
            "texts": texts[:3],
            "max_features": 20
        })
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Успешно! Матрица размером: {data['shape']}")
        else:
            print(f"   ❌ Ошибка: {response.status_code}")
        
        # 3. Тест LSA
        print("\n3. 🔍 Тестируем LSA (из sklearn):")
        response = requests.post(f"{self.base_url}/lsa", json={
            "texts": texts[:5],
            "max_features": 30,
            "n_components": 2
        })
        if response.status_code == 200:
            data = response.json()
            if 'variance' in data:
                print(f"   ✅ Успешно! Объясненная дисперсия: {data['variance']}")
            else:
                print(f"   ✅ Успешно! Ответ получен")
        else:
            print(f"   ❌ Ошибка: {response.status_code}")
        
        # 4. Тест Word2Vec
        print("\n4. 🤖 Тестируем Word2Vec (из sklearn):")
        response = requests.post(f"{self.base_url}/word2vec", json={
            "texts": texts[:5],
            "max_features": 25,
            "n_components": 3
        })
        if response.status_code == 200:
            data = response.json()
            if 'embeddings' in data:
                print(f"   ✅ Успешно! Эмбеддинги: {len(data['embeddings'])} векторов")
        else:
            print(f"   ❌ Ошибка: {response.status_code}")
        
        # 5. Тест NLTK операции
        print("\n5. 🛠️ Тестируем NLTK операции:")
        
        # Токенизация
        print("   • Токенизация (на английском):")
        response = requests.post(f"{self.base_url}/text_nltk/tokenize", 
                               json={"text": "The quick brown fox jumps over the lazy dog while programming in Python."})
        if response.status_code == 200:
            data = response.json()
            print(f"     ✅ {len(data['tokens'])} токенов")
            print(f"     Пример: {data['tokens'][:5]}...")
        
        # Стемминг
        print("   • Стемминг (на английском):")
        response = requests.post(f"{self.base_url}/text_nltk/stem",
                               json={"text": "running jumping laughing programmer studying computers"})
        if response.status_code == 200:
            data = response.json()
            print(f"     ✅ Результат: {data['stems']}")
        
        # Лемматизация
        print("   • Лемматизация (на английском):")
        response = requests.post(f"{self.base_url}/text_nltk/lemmatize",
                               json={"text": "boys were running quickly in beautiful parks with dogs"})
        if response.status_code == 200:
            data = response.json()
            print(f"     ✅ Результат: {data['lemmas']}")
        
        # POS тегинг
        print("   • POS-тегинг (на английском):")
        response = requests.post(f"{self.base_url}/text_nltk/pos_tag",
                               json={"text": "Beautiful cats run fast in green gardens near big cities"})
        if response.status_code == 200:
            data = response.json()
            print(f"     ✅ {len(data['pos_tags'])} тегов")
            print(f"     Пример: {data['pos_tags'][:3]}")
        
        # NER (распознавание сущностей)
        print("   • NER (Распознавание сущностей):")
        response = requests.post(f"{self.base_url}/text_nltk/ner",
                               json={"text": "John Smith works in New York at Google company with Mary Johnson"})
        if response.status_code == 200:
            data = response.json()
            if data['entities']:
                print(f"     ✅ Найдены сущности: {data['entities']}")
            else:
                print("     ℹ️ Сущности не найдены")
        
        print("\n" + "=" * 60)
        print("🌐 ДОПОЛНИТЕЛЬНЫЕ СПОСОБЫ ТЕСТИРОВАНИЯ:")
        print("=" * 60)
        print(f"📖 Swagger UI (интерактивная документация): {self.base_url}/docs")
        print(f"📚 ReDoc (альтернативная документация): {self.base_url}/redoc")
        print(f"🏠 Главная страница API: {self.base_url}/")
        print("=" * 60)
        
        return True

def quick_test():
    """Быстрая проверка работы сервера"""
    client = SimpleNLPClient()
    
    try:
        # Проверяем доступность сервера
        response = requests.get(f"{client.base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Сервер работает! 🎉")
            print(f"📝 Сообщение: {response.json()['message']}\n")
            
            # Запускаем полный тест
            client.test_all_endpoints()
        else:
            print(f"❌ Сервер ответил с ошибкой: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Не могу подключиться к серверу!")
        print("\n🔧 УБЕДИТЕСЬ ЧТО:")
        print("   1. Сервер запущен в другом окне терминала")
        print("   2. Вы выполнили: cd server && python main.py")
        print("   3. Сервер запущен на http://localhost:8000")
        print("\n💡 Команда для запуска сервера:")
        print("   cd server && python -m uvicorn main:app --reload")
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")

if __name__ == "__main__":
    quick_test()