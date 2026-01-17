cd server
pip install -r requirements.txt
# NLP Микросервис на FastAPI

## 🎯 Описание
Веб-сервис для обработки текстов с 9 NLP методами.

## 📋 Функции
- TF-IDF на чистом NumPy
- Bag of Words на чистом NumPy
- LSA (scikit-learn)
- Word2Vec (scikit-learn)
- NLTK: токенизация, стемминг, лемматизация, POS-тегинг, NER

## 🚀 Запуск
```bash
cd server
pip install -r requirements.txt
python main.py

**Когда закончите, нажмите Enter после `EOF`**

Затем продолжайте:

## 📝 СОЗДАЙТЕ .gitignore:

```bash
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
venv/
.env
.DS_Store
*.log
